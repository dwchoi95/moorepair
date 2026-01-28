from ollama import AsyncClient
from pydantic import BaseModel

    
class Ollama:
    def __init__(self, 
                 model:str="codellama:7b", 
                 temperature:float=0.8,
                 token_limit:int=4096,
                 timeout:int=30):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        LOCAL_API_URL = os.getenv("LOCAL_API_URL")
        
        self.client = AsyncClient(host=LOCAL_API_URL, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.token_limit = token_limit
        self.timeout = timeout

    async def run(self, system:str, user:str, format:BaseModel) -> BaseModel| None:
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.token_limit,
                },
                format=format.model_json_schema()
            )
            res = format.model_validate_json(response.message.content)
            return res
        except Exception as e:
            # print(e)
            pass
        return None
        