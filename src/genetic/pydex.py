import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel

from .llm import LLM
from ..execution import Tester
from ..utils import Regularize


class OutFormat(BaseModel):
    correct_program: str

PROGRAM = """[[Buggy Program]]
### Buggy Program ###
{buggy}"""

DIAGNOSTICS = """[[ Diagnostics ]]
{test_results}"""

DESCRIPTION = """[[ Problem Description ]]
{description}"""

TESTSUITE = """[[ Test Suite ]]
{test_cases}"""

OUTPUT = """### Correct Program ###
"""

class PyDex:
    def __init__(self):
        self.objectives = [
            ['program'], 
            ['program', 'diagnostics'], 
            ['program', 'description'], 
            ['program', 'diagnostics', 'description'],
            ['program', 'diagnostics', 'description', 'testsuite'],
            ['program', 'diagnostics', 'testsuite']]
        self.llm = LLM()
    
    def is_valid(self, code:str) -> bool:
        try:
            code = Regularize.run(code)
            if code.strip(): return True
        except SyntaxError: pass
        except Exception as e:
            # print(str(e))
            # print(code)
            # exit(1)
            pass
        return False
    
    def make_prompt(self, buggy:str, comb:list) -> tuple[str, OutFormat]:
        # Prompt
        prompts = []
        for part in comb:
            assert part in ['program', 'diagnostics', 'description', 'testsuite'], f"Invalid part: {part}"
            if part == 'program':
                program = PROGRAM.format(buggy=buggy)
                prompts.append(program)
            elif part == 'diagnostics':
                results = Tester.test(buggy)
                test_results = Tester.print_test_results(results)
                diagnostics = DIAGNOSTICS.format(test_results=test_results)
                prompts.append(diagnostics)
            elif part == 'description':
                description = DESCRIPTION.format(description=Tester.description)
                prompts.append(description)
            elif part == 'testsuite':
                test_cases = str(Tester.testsuite)
                testsuite = TESTSUITE.format(test_cases=test_cases)
                prompts.append(testsuite)
        prompt = "\n\n".join(prompts) + "\n\n" + OUTPUT
        return prompt, OutFormat
    
    def _task(self, buggy:str, comb:list):
        a_async = self.llm.pydex(*self.make_prompt(buggy, comb))
        return a_async
    
    async def __run_async(self, buggy:str) -> str:
        tasks = [asyncio.create_task(self._task(buggy, comb)) for comb in self.objectives]
        
        solutions = []
        pbar = tqdm_async(total=len(tasks), desc='PyDex', leave=False, position=2)
        for coro in asyncio.as_completed(tasks):
            response:OutFormat = await coro
            try:
                solution = self.llm.post_process(response.correct_program)
                if self.is_valid(solution):
                    solutions.append(solution)
            except: continue
            pbar.update(1)
        pbar.close()
        return solutions
    
    def run(self, buggy:str) -> dict:
        return asyncio.run(self.__run_async(buggy))

