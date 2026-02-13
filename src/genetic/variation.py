import re
import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel

from ..execution import Tester, Program
from ..utils import Randoms

class OutFormat(BaseModel):
    fixed_program: str

INITIAL_SYSTEM_PROMPT = '''# Role
You are a program repair assistant in {language} language for student programs.

# Task
Generate the "Fixed Program" by repairing the "Student Program" according to the following guidelines:  
  - Correctness: Pass all test cases.
  - Similarity: Make minimal changes.
  - Efficiency: Optimize execution time and memory usage.
  
# Output
  - Output ONLY the Fixed Program.
  - No explanations, no markdown, no extra comments.
'''

INITIAL_USER_PROMPT = """# Programming Assignment
{assignment_info}

# Student Program
{buggy_program}

# Test Cases
{test_cases}
"""

VARIATION_SYSTEM_PROMPT = '''# Role
You are a program repair assistant in {language} language for student programs.

# Task
Generate a Fixed Program by following the guidelines:  
  1. Crossover the two Peer Programs to combine their strengths.
  2. Mutate to resolve issues (variable conflicts, type/scope mismatches, missing glue code, boundary conditions, etc.) that arise during crossover.
  
# Output
  - Output ONLY the Fixed Program.
  - No explanations, no markdown, no extra comments.
'''

VARIATION_USER_PROMPT = """# Programming Assignment
{assignment_info}

# Peer Program A
{parent1}
## Peer A's Strengths
{parent1_strengths}

# Peer Program B
{parent2}
## Peer B's Strengths
{parent2_strengths}
"""


class Variation:
    def __init__(self, description:str=""):
        self.description = description
    
    def make_prompt(self, buggy:Program, parent:tuple=None) -> tuple[str, str]:
        if parent is None:
            # Initial Population
            system = INITIAL_SYSTEM_PROMPT.format(
                language=buggy.ext.capitalize(),
            )
            # results = Tester.run(buggy)
            # _, failed = Tester.tests_split(results)
            # str_tc = ""
            # test_cases = Randoms.sample(list(failed), 2)
            # str_tc = ''
            # for tc in test_cases:
            #     str_tc += str(tc)
            user = INITIAL_USER_PROMPT.format(
                test_cases=str(Tester.testcases),
                assignment_info=self.description,
                buggy_program=f"```{buggy.ext}\n{buggy.code}\n```"
            )
        else:
            # Crossover & Muation
            system = VARIATION_SYSTEM_PROMPT.format(
                language=buggy.ext.capitalize(),
            )
            
            parent1, parent2, strengths1, strengths2 = parent
            user = VARIATION_USER_PROMPT.format(
                assignment_info=self.description,
                parent1=f"```{parent1.ext}\n{parent1.code}\n```",
                parent2=f"```{parent2.ext}\n{parent2.code}\n```",
                parent1_strengths=strengths1,
                parent2_strengths=strengths2,
            )
        
        # Debugging Logs
        # with open('logs/system.md', 'w') as f:
        #     f.write(system)
        # with open('logs/user.md', 'w') as f:
        #     f.write(user)
        return system, user, OutFormat
    
    async def _task(self, buggy:Program, parent:tuple=None):
        from ..llms import Models
        a_async = await Models.run(*self.make_prompt(buggy, parent))
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
    
    async def __run_async(self, buggy:Program, parents:list) -> list[Program]:
        tasks = [asyncio.create_task(self._task(buggy, parent)) for parent in parents]
        
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
        return fixed_programs
    
    def _asyncio_loop(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    def run(self, buggy:Program, parents:list) -> list[Program]:
        loop = self._asyncio_loop()
        return loop.run_until_complete(self.__run_async(buggy, parents))
        