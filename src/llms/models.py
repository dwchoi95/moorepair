import os
import re
import asyncio
import threading
import warnings

import litellm

# Drop provider-unsupported params (e.g., temperature on some reasoning
# models) instead of raising, and keep logs quiet inside the search loop
litellm.drop_params = True
litellm.suppress_debug_info = True
# LiteLLM emits harmless pydantic serializer warnings for some providers
warnings.filterwarnings("ignore", module="pydantic")


class Models:
    """Unified async LLM interface backed by LiteLLM.

    The model name selects the provider by prefix (LiteLLM convention):
      gpt-3.5-turbo, gpt-5-nano        -> OpenAI (OPENAI_API_KEY)
      ollama/codellama, ollama/gemma2  -> local Ollama server
      hosted_vllm/<repo>               -> vLLM OpenAI-compatible server (--api-base)
      vllm/<repo>                      -> vLLM offline engine, NO server needed
                                          (model loads in-process once per run)
      gemini/gemma-3-27b-it            -> Google AI Studio (GEMINI_API_KEY)
      huggingface/<repo>               -> HF Inference (HF_TOKEN)
    """

    # vLLM offline engine state (vllm/<repo> models only)
    _vllm_engine = None
    _vllm_lock = None
    VLLM_MAX_TOKENS = 4096  # SamplingParams default is 16 — far too small

    @classmethod
    def set(cls,
        model:str="gpt-3.5-turbo",
        temperature:float=0.8,
        timeout:int=60,
        api_base:str=None,
    ):
        from dotenv import load_dotenv
        load_dotenv()  # provider keys: OPENAI_API_KEY, GEMINI_API_KEY, HF_TOKEN, ...

        cls.model = model
        if model.startswith("gpt-5"):
            temperature = 1.0
        cls.temperature = temperature
        cls.timeout = timeout
        # Custom endpoint for self-hosted servers (Ollama/vLLM/TGI);
        # falls back to the LLM_API_BASE env var
        cls.api_base = api_base or os.getenv("LLM_API_BASE")

        cls._vllm_engine = None
        if model.startswith("vllm/"):
            cls._init_vllm(model[len("vllm/"):])

    # ---------------- vLLM offline backend ------------------------- #

    @classmethod
    def _init_vllm(cls, repo: str):
        """Load the vLLM offline engine once; reused by every call."""
        from vllm import LLM
        cls._vllm_engine = LLM(model=repo)
        cls._vllm_lock = threading.Lock()

    @classmethod
    def _vllm_chat(cls, system: str, user: str) -> str | None:
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=cls.temperature,
            max_tokens=cls.VLLM_MAX_TOKENS,
        )
        messages = [
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ]
        # The offline engine is not thread-safe; serialize calls
        with cls._vllm_lock:
            try:
                outs = cls._vllm_engine.chat(messages, params, use_tqdm=False)
            except Exception:
                # Some chat templates (e.g., Gemma family) reject the
                # system role; fold it into the user message instead
                merged = [{ "role": "user", "content": f"{system}\n\n{user}" }]
                outs = cls._vllm_engine.chat(merged, params, use_tqdm=False)
        return outs[0].outputs[0].text

    @staticmethod
    def _parses(code: str) -> bool:
        import ast
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    @classmethod
    def _post_process(cls, text: str) -> str:
        """Extract program code from an LLM response.

        Handles both response styles: the whole message being one code
        block (GPT-style) and prose with embedded code blocks anywhere
        (CodeLlama/Gemma-style). Candidates are every fenced block —
        including an unterminated final block from a cut-off response —
        plus the bare text itself; the longest Python-parseable
        candidate wins. If nothing parses, the longest block (or the
        raw text) is returned and discarded by the caller's syntax check."""
        text = text.strip()

        blocks = re.findall(
            r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n(.*?)(?:```|\Z)",
            text,
            flags=re.DOTALL,
        )
        candidates = [b.strip() for b in blocks if b.strip()]
        candidates.append(text)

        parseable = [c for c in candidates if cls._parses(c)]
        if parseable:
            return max(parseable, key=len)
        return candidates[0] if blocks else text

    # Single retry policy shared by ALL approaches (MooRepair and the
    # baseline alike): up to 3 attempts per LLM call on failure
    MAX_RETRIES = 3

    @classmethod
    async def run(cls, system:str, user:str) -> str | None:
        for _ in range(cls.MAX_RETRIES):
            try:
                if cls._vllm_engine is not None:
                    # Offline vLLM: blocking engine call off the event loop
                    content = await asyncio.to_thread(cls._vllm_chat, system, user)
                else:
                    response = await litellm.acompletion(
                        model=cls.model,
                        messages=[
                            { "role": "system", "content": system },
                            { "role": "user", "content": user }
                        ],
                        temperature=cls.temperature,
                        timeout=cls.timeout,
                        api_base=cls.api_base,
                    )
                    content = response.choices[0].message.content
                if content is None:
                    return None
                return cls._post_process(content)
            except Exception as e:
                # print(e) # DEBUG
                continue
        return None
