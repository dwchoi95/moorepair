import re
from openai import AsyncOpenAI
from pydantic import BaseModel


class OutFormat(BaseModel):
    patch: str

class Models:
    @classmethod
    def set(cls,
        model:str="codellama/CodeLlama-7b-Instruct-hf", 
        temperature:float=0.8,
        timeout:int=60
    ):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        LOCAL_API_URL = os.getenv("LOCAL_API_URL")
        API_KEY = os.getenv("OPENAI_API_KEY")
        
        cls.client = AsyncOpenAI(api_key=API_KEY, base_url=LOCAL_API_URL)
        cls.model = model
        cls.temperature = temperature
        cls.timeout = timeout
    
    @classmethod
    def _post_process(cls, code: str) -> str:
        code = code.strip()
        while code.startswith("```") and code.endswith("```"):
            m = re.search(r"```(?:[a-zA-Z0-9_+-]+)?[\r\n]+(.*?)```", code, flags=re.DOTALL)
            if m:
                code = m.group(1).strip()
            else:
                break
        return code
    
    @classmethod
    async def run(cls, system:str, user:str) -> str | None:
        try:
            response = await cls.client.chat.completions.parse(
                model=cls.model, 
                messages=[
                    { "role": "system", "content": system },
                    { "role": "user", "content": user }
                ],
                temperature=cls.temperature,
                timeout=cls.timeout,
                response_format=OutFormat,
            )
            content = response.choices[0].message
            if content.parsed:
                return cls._post_process(content.parsed.patch)
            return None
        except Exception as e:
            # print(e) # DEBUG
            pass
        return None