import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel

from ..llms import OpenAI, Ollama
from ..execution import Tester, Program, Programs

class OutFormat(BaseModel):
    fixed_program: str

SYSTEM_PROMPT = '''# Identity

You are an expert in program repair systems specializing in genetic programming mutation.


# Instructions

You will be given a 'Problem Description', a 'Buggy Program' and a set of 'Test Results' as Inputs.
Mutate the code of the given 'Buggy Program' according to the following guideline:
{guideline}

Outputs should be a 'Fixed Program' as str.
'''

USER_PROMPT = """# Inputs


## Problem Description
{description}


## Buggy Program
```python
{buggy}
```

## Test Results
{test_results}
"""



class Mutation:
    def __init__(self, model:OpenAI|Ollama, description:str=""):
        self.model = model
        self.description = description
        
    def make_prompt(self, buggy:Program, guideline:str) -> tuple[str, str]:
        # System Prompt
        system = SYSTEM_PROMPT.format(guideline=guideline)
        
        # User Prompt
        Tester.run(buggy.code, buggy.ext)
        user = USER_PROMPT.format(
            description=self.description,
            buggy=buggy.code,
            test_results=str(buggy.results)
        )
        return system, user, OutFormat
    
    def _task(self, buggy:Program, guideline:str):
        a_async = self.model.run(*self.make_prompt(buggy, guideline))
        return a_async, buggy.ext
    
    async def __run_async(self, programs:list) -> Programs:
        tasks = [asyncio.create_task(self._task(buggy, guideline)) for (buggy, guideline) in programs]
        
        fixed_programs = []
        pbar = tqdm_async(total=len(tasks), desc='Mutation', leave=False, position=2)
        for coro, extension in asyncio.as_completed(tasks):
            code:OutFormat = await coro
            fixed_programs.append(Program(
                id=f"mutate_{len(fixed_programs)+1}",
                code=code.fixed_program,
                ext=extension
            ))
            pbar.update(1)
        pbar.close()
        return Programs(fixed_programs)
    
    def run(self, programs:list) -> Programs:
        return asyncio.run(self.__run_async(programs))