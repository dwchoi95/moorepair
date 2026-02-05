import re
import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel

from .fitness import Fitness
from ..execution import Tester, Program, Programs
from ..utils import Randoms


class OutFormat(BaseModel):
    fixed_program: str

SYSTEM_PROMPT = '''# Role

You are an expert programming tutor in {language} who helps to fix buggy program.

# Task

As Inputs, you will receive a "Buggy Program", a "Reference Program", their test results, and a "Problem Description", a set of "Test Cases".
Generate the "Fixed Program" by repairing the "Buggy Program" according to the following priority guidelines:
{guidelines}

The Output should be a "Fixed Program" as string.
'''

USER_PROMPT = """# Inputs

## Buggy Program
{buggy_program}
### Buggy Test Results
{buggy_results}

## Reference Program
{reference_program}
### Reference Test Results
{reference_results}

## Problem Description
{description}

## Test Cases
{test_cases}
"""


class Variation:
    def __init__(self, 
                 fitness:Fitness,
                 description:str=""):
        self.fitness = fitness
        self.description = description
        
    def prioritization(self, buggy:Program, references:Programs):
        # 목적함수별 fitness 점수로 프로그램별 목적함수 우선순위 매기기
        normalized = self.fitness.run(buggy, references)
        scores = {obj: {refer.id: normalized[refer.id][obj] for refer in references}
                   for obj in self.fitness.objectives}
        # obj별로 score가 낮은 순서대로 정렬하여 순위 매기기
        rankings = {obj: sorted(scores[obj], key=lambda x: scores[obj][x]) 
                    for obj in self.fitness.objectives}
        # 프로그램별로 각 목적함수에서의 순위가 높은 순서대로 정렬하여 개별 program.meta['priorities']에 저장
        for refer in references:
            refer.meta['priorities'] = sorted(
                self.fitness.objectives,
                key=lambda obj: rankings[obj].index(refer.id)
            )
        return references
    
    def make_prompt(self, buggy:Program, refer:Program) -> tuple[str, str]:
        from ..llms import Tokenizer, Models
        
        priorities = refer.meta.get('priorities')
        guidelines = ''
        for i, obj in enumerate(priorities, 1):
            guidelines += f"  {i}. {self.fitness.guidelines[obj]}\n"
        
        # System Prompt
        system = SYSTEM_PROMPT.format(
            language=buggy.ext.capitalize(),
            guidelines=guidelines
        )
        
        b_passed, b_failed = Tester.tests_split(buggy.results)
        r_passed, r_failed = Tester.tests_split(refer.results)
        
        buggy_results = f"Passed Test Cases IDs: {sorted([tc.id for tc in b_passed])}  \n"
        buggy_results += f"Failed Test Cases IDs: {sorted([tc.id for tc in b_failed])}\n"
        reference_results = f"Passed Test Cases IDs: {sorted([tc.id for tc in r_passed])}  \n"
        reference_results += f"Failed Test Cases IDs: {sorted([tc.id for tc in r_failed])}\n"
        
        # User Prompt
        user = USER_PROMPT.format(
            description=self.description,
            test_cases="",
            buggy_program=f"```{buggy.ext}\n{buggy.code}\n```",
            buggy_results=buggy_results,
            reference_program=f"```{refer.ext}\n{refer.code}\n```",
            reference_results=reference_results,
        )
        
        token_limit = Models.token_limit
        base_tokens = Tokenizer.length(system + user)
        
        pass2pass = b_passed.intersection(r_passed)
        fail2pass = b_failed.intersection(r_passed)
        union = pass2pass.union(fail2pass)
        union_list = list(union)
        Randoms.shuffle(union_list)
        
        testcases = []
        b_pass_filter, b_fail_filter = set(), set()
        r_pass_filter, r_fail_filter = set(), set()
        for tc in union_list:
            test_case = str(tc)
            testcases.append(test_case)
            tokens = Tokenizer.length("\n".join(testcases))
            if base_tokens + tokens > token_limit:
                break
            if tc in b_passed:
                b_pass_filter.add(tc)
            if tc in b_failed:
                b_fail_filter.add(tc)
            if tc in r_passed:
                r_pass_filter.add(tc)
            if tc in r_failed:
                r_fail_filter.add(tc)
        
        buggy_results = f"Passed Test Cases IDs: {sorted([tc.id for tc in b_pass_filter])}  \n"
        buggy_results += f"Failed Test Cases IDs: {sorted([tc.id for tc in b_fail_filter])}\n"
        reference_results = f"Passed Test Cases IDs: {sorted([tc.id for tc in r_pass_filter])}  \n"
        reference_results += f"Failed Test Cases IDs: {sorted([tc.id for tc in r_fail_filter])}\n"
        
        user = USER_PROMPT.format(
            description=self.description,
            test_cases="\n".join(testcases),
            buggy_program=f"```{buggy.ext}\n{buggy.code}\n```",
            buggy_results=buggy_results,
            reference_program=f"```{refer.ext}\n{refer.code}\n```",
            reference_results=reference_results,
        )
        
        # Debugging Logs
        # with open('logs/system.md', 'w') as f:
        #     f.write(system)
        # with open('logs/user.md', 'w') as f:
        #     f.write(user)
        return system, user, OutFormat
    
    async def _task(self, buggy:Program, refer:Program):
        from ..llms import Models
        a_async = await Models.run(*self.make_prompt(buggy, refer))
        return a_async
    
    def _post_process(self, code:str) -> str:
        code = code.strip()
        while code.startswith("```") and code.endswith("```"):
            m = re.search(r"```(?:[a-zA-Z0-9_+-]+)?[\r\n]+(.*?)```", code, flags=re.DOTALL)
            if m:
                code = m.group(1).strip()
            else:
                break
        return code
    
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
                id=f"child_{len(fixed_programs)+1}",
                code=self._post_process(response.fixed_program),
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