import re
from openai import AsyncOpenAI

class Models:
    @classmethod
    def set(cls,
        model:str="gpt-3.5-turbo", 
        temperature:float=0.8,
        timeout:int=60
    ):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        API_KEY = os.getenv("OPENAI_API_KEY")
        
        cls.client = AsyncOpenAI(api_key=API_KEY, timeout=timeout)
        cls.model = model
        if model.startswith("gpt-5"):
            temperature = 1.0
        cls.temperature = temperature
        cls.timeout = timeout
    
    @classmethod
    def _post_process(cls, code: str) -> str:
        code = code.strip()
        
        while code.startswith("```") and code.endswith("```"):
            m = re.search(
                r"```(?:[a-zA-Z0-9_+-]+)?[\r\n]+(.*?)```",
                code,
                flags=re.DOTALL,
            )
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
            )
            content = response.choices[0].message.content
            return cls._post_process(content)
        except Exception as e:
            # print(e) # DEBUG
            pass
        return None
        