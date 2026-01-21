from ollama import AsyncClient
from pydantic import BaseModel

    
class Ollama:
    def __init__(self, 
                 model:str="llama3.1:8b", 
                 temperature:float=0.0):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        LOCAL_API_URL = os.getenv("LOCAL_API_URL")
        
        self.host = LOCAL_API_URL
        self.model = model
        self.temperature = temperature
        self.client = AsyncClient(host=LOCAL_API_URL)

    async def run(self, system:str, user:str, format:BaseModel) -> str| bool:
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={
                    "temperature": self.temperature,
                },
                format=format.model_json_schema()
            )
            res = format.model_validate_json(response.message.content)
            return res
        except Exception as e:
            print(e)
            pass
        return None
        