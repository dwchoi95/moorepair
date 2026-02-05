from openai import AsyncOpenAI
from pydantic import BaseModel


class Models:
    @classmethod
    def set(cls,
                 model:str="codellama/CodeLlama-7b-Instruct-hf", 
                 temperature:float=0.8,
                 token_limit:int=4096,
                 timeout:int=10):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        LOCAL_API_URL = os.getenv("LOCAL_API_URL")
        
        cls.client = AsyncOpenAI(
            timeout=timeout, 
            base_url=LOCAL_API_URL,
            max_retries=0)
        cls.model = model
        cls.temperature = temperature
        cls.token_limit = token_limit
        cls.timeout = timeout
    
    @classmethod
    async def run(cls, system:str, user:str, format:BaseModel):
        try:
            response = await cls.client.chat.completions.create(
                model=cls.model, 
                messages=[
                    { "role": "system", "content": system },
                    { "role": "user", "content": user }
                ],
                temperature=cls.temperature,
                max_tokens=cls.token_limit,
                timeout=cls.timeout,
                extra_body={
                    "structured_outputs": {
                        "json": format.model_json_schema(),
                    }
                }
            )
            content = response.choices[0].message.content
            res = format.model_validate_json(content)
            return res
        except Exception as e:
            # print(e)
            pass
        return None