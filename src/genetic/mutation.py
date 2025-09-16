import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel

from .llm import LLM
from ..execution import Tester

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
    def __init__(self):
        self.llm = LLM()
        
    def make_prompt(self, buggy:str, guideline:str) -> tuple[str, str]:
        # System Prompt
        system = SYSTEM_PROMPT.format(guideline=guideline)
        
        # User Prompt
        results = Tester.test(buggy)
        test_results = Tester.print_test_results(results)
        user = USER_PROMPT.format(
            description=Tester.description,
            buggy=buggy,
            test_results=test_results
        )
        return system, user, OutFormat
    
    def _task(self, buggy:str, guideline:str):
        a_async = self.llm.run(*self.make_prompt(buggy, guideline))
        return a_async
    
    async def __run_async(self, programs:list) -> str:
        tasks = [asyncio.create_task(self._task(buggy, guideline)) for (buggy, guideline) in programs]
        
        fixed_programs = []
        pbar = tqdm_async(total=len(tasks), desc='Mutation', leave=False, position=2)
        for coro in asyncio.as_completed(tasks):
            response:OutFormat = await coro
            try:
                program = self.llm.post_process(response.fixed_program)
                fixed_programs.append(program)
            except: continue
            pbar.update(1)
        pbar.close()
        return fixed_programs
    
    def run(self, programs:list) -> list:
        return asyncio.run(self.__run_async(programs))