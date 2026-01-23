import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel

from ..llms import OpenAI, Ollama
from ..execution import Tester, Programs, Program

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
    def __init__(self, model:OpenAI|Ollama, description:str=""):
        self.model = model
        self.description = description
    
    def make_prompt(self, programs:Programs) -> tuple[str, str]:
        # System Prompt
        system = SYSTEM_PROMPT.format(pop_size=len(programs))
        
        # User Prompt
        user = USER_PROMPT.format(
            description=self.description,
            test_cases=str(Tester.testcases),
            programs="\n\n".join([f"### Buggy Program {i}\n```{p.ext}\n{p.code}\n```" 
                                  for i, p in enumerate(programs, start=1)])
        )
        return system, user, OutFormat
    
    async def _task(self, programs:Programs):
        a_async = await self.model.run(*self.make_prompt(programs))
        return a_async
    
    async def __run_async(self, programs:Programs) -> Programs:
        # Run single task
        offsprings = []
        pbar = tqdm_async(total=len(programs), desc='Crossover', leave=False, position=2)
        response: OutFormat = await self._task(programs)
        try: results = response.offsprings
        except: results = []
        for prog, patch in zip(programs, results):
            offsprings.append(Program(
                id=f"cross_{len(offsprings)+1}",
                code=patch,
                ext=prog.ext
            ))
            pbar.update(1)
        pbar.close()
        return Programs(offsprings)
    
    # def run(self, programs:Programs) -> Programs:
    #     return asyncio.run(self.__run_async(programs))
    
    def run(self, programs:list) -> Programs:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.__run_async(programs))