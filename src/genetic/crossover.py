import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel

from .llm import LLM
from ..execution import Tester

class OutFormat(BaseModel):
    offsprings: list[str]

SYSTEM_PROMPT = '''# Identity

You are an expert in program synthesis systems specializing in genetic programming crossover.


# Instructions

You will be given a 'Problem Description', a set of 'Test Cases', and 'Buggy Programs' as Inputs.
Crossover the code of the given 'Buggy Programs' according to the following guidelines:
  - Use only the given code of 'Buggy Programs'
  - Make sure there are no compilation errors

Outputs should be a {pop_size} 'Fixed Programs' as List[str].
'''

USER_PROMPT = """# Inputs

## Problem Description
{description}

## Test Cases
{test_cases}

## Buggy Programs
{programs}
"""

class Crossover:
    def __init__(self):
        self.llm = LLM()
    
    def make_prompt(self, programs:list) -> tuple[str, str]:
        # System Prompt
        system = SYSTEM_PROMPT.format(pop_size=len(programs))
        
        # User Prompt
        user = USER_PROMPT.format(
            description=Tester.description,
            test_cases=str(Tester.testsuite),
            programs="\n\n".join([f"### Buggy Program {i}\n```python\n{p}\n```" 
                                  for i, p in enumerate(programs, start=1)])
        )
        return system, user, OutFormat
    
    def _task(self, programs:list):
        a_async = self.llm.run(*self.make_prompt(programs))
        return a_async
    
    async def __run_async(self, programs:list) -> str:
        # Run single task
        offsprings = []
        pbar = tqdm_async(total=len(programs), desc='Crossover', leave=False, position=2)
        response: OutFormat = await self._task(programs)
        try: results = response.offsprings
        except: results = []
        for program in results:
            program = self.llm.post_process(program)
            offsprings.append(program)
            pbar.update(1)
        pbar.close()
        return offsprings
    
    def run(self, programs:list) -> list:
        return asyncio.run(self.__run_async(programs))