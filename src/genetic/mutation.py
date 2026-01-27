import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel
import math

from .fitness import Fitness
from ..execution import Tester, Program, Programs
from ..utils import Randoms

class OutFormat(BaseModel):
    fixed_program: str

SYSTEM_PROMPT = '''# Role

You are an expert programming tutor who helps to improve fixed program by mutating their code.

# Task

You will be given a 'Buggy Program', 'Problem Description', a set of 'Test Cases' and a 'Fixed Program' as Inputs.
Generate an 'Improved Fixed Program' by mutating the code of the given 'Fixed Program' according to the following guidelines:
  - Start from the given 'Fixed Program' as the base code
  - 'Improved Fixed Program' must pass all 'Test Cases'
{guideline}

Outputs should be a 'Improved Fixed Program' as str.
'''

USER_PROMPT = """# Inputs

## Buggy Program
{buggy_program}

## Problem Description
{description}

## Test Cases
{test_cases}

## Fixed Program
{fixed_program}
"""



class Mutation:
    def __init__(self, fitness:Fitness, description:str=""):
        self.fitness = fitness
        self.description = description
    
    def stochastic_universal_sampling(self, evaluates:dict) -> list:
        if not evaluates:
            raise ValueError("evaluates must not be empty")

        EPS = 1e-12
        items = []
        min_w = float('inf')

        for k, v in evaluates.items():
            base = float(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0
            w = 1.0 - base
            if math.isclose(w, 0.0, abs_tol=EPS) or math.isclose(base, 1.0, abs_tol=EPS):
                continue
            items.append((k, w))
            if w < min_w:
                min_w = w

        if not items:
            raise ValueError("No eligible items after excluding entries with v == 1.0 (or 1 - v == 0).")

        if min_w < 0.0:
            shift = -min_w
            items = [(k, w + shift) for k, w in items]

        total = sum(w for _, w in items)
        if total <= 0.0:
            return Randoms.choice([k for k, _ in items])

        pointer = Randoms.random() * total
        acc = 0.0
        for k, w in items:
            acc += w
            if acc >= pointer:
                return k
        return items[-1][0]
    
    def make_prompt(self, buggy:Program, fixed:Program) -> tuple[str, str]:
        evaluates = self.fitness.run(buggy, fixed)
        obj = self.stochastic_universal_sampling(evaluates)
        guideline = self.fitness.GUIDELINES.get(obj, None)
        if guideline is None:
            guideline = Randoms.choice(list(self.fitness.GUIDELINES.values()))
        
        # System Prompt
        system = SYSTEM_PROMPT.format(guideline=guideline)
        
        # User Prompt
        user = USER_PROMPT.format(
            description=self.description,
            buggy_program=f"```{buggy.ext}\n{buggy.code}\n```",
            test_cases=str(Tester.testcases),
            fixed_program=f"```{fixed.ext}\n{fixed.code}\n```"
        )
        return system, user, OutFormat
    
    async def _task(self, buggy:Program, fixed:Program):
        from ..llms import Spec
        a_async = await Spec.model.run(*self.make_prompt(buggy, fixed))
        return a_async
    
    async def __run_async(self, buggy:Program, programs:list) -> Programs:
        tasks = [asyncio.create_task(self._task(buggy, fixed)) for fixed in programs]
        
        fixed_programs = []
        pbar = tqdm_async(total=len(tasks), desc='Mutation', leave=False, position=2)
        for coro in asyncio.as_completed(tasks):
            response: OutFormat = await coro
            pbar.update(1)
            if response is None: continue
            fixed_programs.append(Program(
                id=f"mutate_{len(fixed_programs)+1}",
                code=response.fixed_program,
                ext=buggy.ext
            ))
        pbar.close()
        return Programs(fixed_programs)
    
    # def run(self, programs:list) -> Programs:
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