import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel

from .fitness import Fitness
from ..execution import Tester, Programs, Program
from ..utils import Randoms


class OutFormat(BaseModel):
    offsprings: list[str]

SYSTEM_PROMPT = '''# Role

You are an expert programming tutor who helps to fix buggy program by combining code snippets from reference programs.

# Task

You will be given a 'Buggy Program', 'Problem Description', a set of 'Test Cases', and 'Reference Programs' as Inputs.
Generate two 'Fixed Programs' by crossover the code snippets of the given 'Reference Programs' according to the following guidelines:
  - Use only the given code of 'Reference Programs'
  - 'Fixed Program' must pass all 'Test Cases'
{guidelines}

# Output Format
Outputs should be a list of 'Fixed Programs' as list[str].
'''

USER_PROMPT = """# Inputs

## Buggy Program
{buggy_program}

## Problem Description
{description}

## Test Cases
{test_cases}

## Reference Programs
### Reference Program 1
{reference_program_1}

### Reference Program 2
{reference_program_2}
"""

class Crossover:
    def __init__(self, fitness:Fitness, description:str=""):
        self.fitness = fitness
        self.description = description
    
    def make_pairs(self, buggy:Program, programs:Programs) -> Programs:
        pairs = Programs()
        shuffled = list(programs)
        Randoms.shuffle(shuffled)
        for i in range(0, len(shuffled)-1, 2):
            p1 = self.fitness.run(buggy, shuffled[i])
            p2 = self.fitness.run(buggy, shuffled[i+1])
            better = sum(1 for obj in self.fitness.OBJECTIVES if p1.get(obj, 0) < p2.get(obj, 0))
            if better == 0 or better == len(self.fitness.OBJECTIVES):
                continue
            pairs.append(shuffled[i])
            pairs.append(shuffled[i+1])
        return pairs

    def make_prompt(self, buggy:Program, refer_1:Program, refer_2:Program):
        # System Prompt
        system = SYSTEM_PROMPT.format(guidelines="\n".join(
            list(self.fitness.guidelines.values())))
        
        # User Prompt
        user = USER_PROMPT.format(
            buggy_program=f"```{buggy.ext}\n{buggy.code}\n```",
            description=self.description,
            test_cases=str(Tester.testcases),
            reference_program_1=f"```{refer_1.ext}\n{refer_1.code}\n```",
            reference_program_2=f"```{refer_2.ext}\n{refer_2.code}\n```"
        )
        return system, user, OutFormat
    
    async def _task(self, buggy:Program, refer_1:Program, refer_2:Program):
        from ..llms import Spec
        a_async = await Spec.model.run(*self.make_prompt(buggy, refer_1, refer_2))
        return a_async, refer_1, refer_2

    async def __run_async(self, buggy:Program, programs:Programs) -> Programs:
        pairs = self.make_pairs(buggy, programs)
        tasks = [asyncio.create_task(self._task(buggy, pairs[i], pairs[i+1])) 
                 for i in range(0, len(pairs)-1, 2)]
        
        offsprings = [] if len(pairs) % 2 == 0 else [pairs[-1]]
        pbar = tqdm_async(total=len(pairs), desc='Crossover', leave=False, position=2)
        for coro in asyncio.as_completed(tasks):
            response, refer_1, refer_2 = await coro
            pbar.update(1)
            if response is None: continue
            for offs in response.offsprings:
                offsprings.append(Program(
                    id=f"cross_{len(offsprings)+1}",
                    code=offs,
                    ext=buggy.ext,
                    meta={"parent1": refer_1.id, "parent2": refer_2.id}
                ))
        pbar.close()
        return Programs(offsprings)
    
    # def run(self, programs:Programs) -> Programs:
    #     return asyncio.run(self.__run_async(programs))
    
    def run(self, buggy:Program, programs:Programs) -> Programs:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.__run_async(buggy, programs))