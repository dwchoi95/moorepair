import re
import time
from openai import (
    OpenAI,
    AsyncOpenAI, 
    RateLimitError, 
    APIError, 
    APIConnectionError, 
    AuthenticationError
)
from pydantic import BaseModel


class LLM:
    def __init__(self,
                 model:str="gpt-5-nano",
                 temperature:float=1.0):
        self.client = AsyncOpenAI()
        self.model = model
        self.temperature = temperature
    
    def post_process(self, code:str) -> str:
        # Post-process the code
        py_pattern = r'```(?:[Pp]ython\s*)?([\s\S]*?)```'
        xml_pattern = r'<fixed_program>([\s\S]*?)</fixed_program>'
        if code and re.search(py_pattern, code):
            code = re.findall(py_pattern, code, flags=re.DOTALL)[0]
        elif code and re.search(xml_pattern, code):
            code = re.findall(xml_pattern, code, flags=re.DOTALL)[0]
        return code
    
    async def run(self, system:str, user:str, format:BaseModel):
        try:
            response = await self.client.responses.parse(
                model=self.model, 
                input=[
                    { "role": "system", "content": system },
                    { "role": "user", "content": user }
                ],
                temperature=self.temperature,
                text_format=format,
            )
            model = response.output_parsed
            return model
        except (RateLimitError, AuthenticationError) as e:
            print(f"OPENAI Error: {str(e)}.\nConfirm your API key and model name.")
            exit()
        except (APIError, APIConnectionError) as e:
            # print(f"OPENAI Error: {str(e)}.\nRetry after a while.")
            time.sleep(3)
            return self.run(system, user, format)
        except Exception as e:
            print(e)
            # print(system)
            # print(user)
            time.sleep(3)
            return self.run(system, user, format)
        return None
    
    async def pydex(self, user:str, format:BaseModel):
        try:
            response = await self.client.responses.parse(
                model=self.model, 
                input=[
                    { "role": "user", "content": user }
                ],
                temperature=self.temperature,
                text_format=format,
            )
            model = response.output_parsed
            return model
        except (RateLimitError, AuthenticationError) as e:
            print(f"OPENAI Error: {str(e)}.\nConfirm your API key and model name.")
            exit()
        except (APIError, APIConnectionError) as e:
            # print(f"OPENAI Error: {str(e)}.\nRetry after a while.")
            time.sleep(3)
            return self.pydex(user, format)
        except Exception as e:
            print(e)
            # print(user)
            time.sleep(3)
            return self.pydex(user, format)
        return None