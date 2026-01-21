import re
from openai import (AsyncOpenAI, RateLimitError, AuthenticationError)
from pydantic import BaseModel

from ..genetic import Fitness


SYSTEM_PROMPT = """# Identity

You are a program repair system that helps to fix the bug and improve performance of program in Python3 code.


# Instructions

You will be given a 'Buggy Program' along with a 'Problem Description' of program, a set of 'Test Results' for all test cases, static analysis 'Warning Messages', and optionally a 'Reference Program' to fix it.
Use all of this information to fix the program, following the fix guidelines:

* Guidelines for Repair:
  - Pass all test cases
{objectives}
* Print in given output format
"""


BASE_PROMPT = """# Inputs

## Buggy Program

```python
{buggy_program}
```

## Description
{description}

## Test Results
{test_results}

## Warning Messages
{warning_messages}
"""

OUTPUT_PROMPT = """

# Output

## Fixed Program"""

USER_PROMPT_NONE = BASE_PROMPT + OUTPUT_PROMPT

USER_PROMPT_REFER = BASE_PROMPT + """
## Reference Program

```python
{reference_program}
```""" + OUTPUT_PROMPT


class OpenAI:
    def __init__(self,
                 model:str="gpt-3.5-turbo", 
                 temperature:float=0.0,
                 timeout:int=10,
                 objectives:list=Fitness.OBJECTIVES):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
    
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
            print(e)
            exit()
        except Exception as e:
            print(e)
            pass
        return None