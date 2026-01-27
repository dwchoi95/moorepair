import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel

from .fitness import Fitness
from ..execution import Tester, Program, Programs
from ..utils import Randoms


class OutFormat(BaseModel):
    fixed_program: str

SYSTEM_PROMPT = '''# Role

You are an expert {language} programming tutor who helps to fix buggy program.

# Task

You will be given a 'Problem Description', a set of 'Test Cases', a 'Buggy Program', a 'Reference Program' and their test results as Inputs.
Generate the 'Fixed Program' by repairing the 'Buggy Program' according to the following priority guidelines:
{guidelines}

Outputs should be a 'Fixed Program' as str.
'''

USER_PROMPT = """# Inputs

## Problem Description
{description}

## Test Cases
{test_cases}

## Buggy Program
{buggy_program}
### Buggy Test Results
{buggy_results}

## Reference Program
{reference_program}
### Reference Test Results
{reference_results}
"""


class Variation:
    def __init__(self, 
                 fitness:Fitness,
                 description:str=""):
        self.fitness = fitness
        self.description = description
        
    def prioritization(self, buggy:Program, references:Programs):
        # 목적함수별 fitness 점수로 프로그램별 목적함수 우선순위 매기기
        scores = {obj: {} for obj in self.fitness.objectives}
        for refer in references:
            fit = self.fitness.run(buggy, refer)
            for obj, score in fit.items():
                scores[obj][refer] = score
        # obj별로 score가 낮은 순서대로 정렬하여 순위 매기기
        rankings = {obj: sorted(scores[obj], key=lambda x: scores[obj][x]) 
                    for obj in self.fitness.objectives}
        # 프로그램별로 각 목적함수에서의 순위가 높은 순서대로 정렬하여 개별 program.meta['priorities']에 저장
        for refer in references:
            refer.meta['priorities'] = sorted(
                self.fitness.objectives,
                key=lambda obj: rankings[obj].index(refer)
            )
        return references
                
    
    def make_prompt(self, buggy:Program, refer:Program) -> tuple[str, str]:
        priorities = refer.meta.get('priorities')
        guidelines = ''
        for i, obj in enumerate(priorities, 1):
            guidelines += f"  {i}. {self.fitness.guidelines[obj]}\n"
            
        b_passed, b_failed = Tester.tests_split(buggy.results)
        r_passed, r_failed = Tester.tests_split(refer.results)
        
        buggy_results = f"Passed Test Cases IDs: {sorted([tc.id for tc in b_passed])}  \n"
        buggy_results += f"Failed Test Cases IDs: {sorted([tc.id for tc in b_failed])}\n"
        reference_results = f"Passed Test Cases IDs: {sorted([tc.id for tc in r_passed])}  \n"
        reference_results += f"Failed Test Cases IDs: {sorted([tc.id for tc in r_failed])}\n"
        
        # System Prompt
        system = SYSTEM_PROMPT.format(
            language=buggy.ext,
            guidelines=guidelines
        )
        
        # User Prompt
        user = USER_PROMPT.format(
            description=self.description,
            test_cases=str(Tester.testcases),
            buggy_program=f"```{buggy.ext}\n{buggy.code}\n```",
            buggy_results=buggy_results,
            reference_program=f"```{refer.ext}\n{refer.code}\n```",
            reference_results=reference_results,
        )
        # with open('logs/system.md', 'w') as f:
        #     f.write(system)
        # with open('logs/user.md', 'w') as f:
        #     f.write(user)
        return system, user, OutFormat
    
    async def _task(self, buggy:Program, refer:Program):
        from ..llms import Spec
        a_async = await Spec.model.run(*self.make_prompt(buggy, refer))
        return a_async
    
    async def __run_async(self, buggy:Program, references:Programs) -> Programs:
        references = self.prioritization(buggy, references)
        tasks = [asyncio.create_task(self._task(buggy, refer)) for refer in references]
        
        fixed_programs = []
        pbar = tqdm_async(total=len(tasks), desc='Variation', leave=False, position=2)
        for coro in asyncio.as_completed(tasks):
            response: OutFormat = await coro
            pbar.update(1)
            if response is None: continue
            fixed_programs.append(Program(
                id=f"pop_{len(fixed_programs)+1}",
                code=response.fixed_program,
                ext=buggy.ext
            ))
        pbar.close()
        return Programs(fixed_programs)
    
    def run(self, buggy:Program, references:Programs) -> Programs:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.__run_async(buggy, references))