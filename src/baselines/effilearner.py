import logging
from pathlib import Path
from tqdm import tqdm
from src.llms import Models
from src.execution import Programs, Program, Tester, Status
from src.utils import ETC

SYSTEM = '''Optimize the efficiency of the following Python code based on the task, test case, and overhead analysis provided. Ensure the optimized code can pass the given test case.'''

USER = '''\
Task Description:
{description}

Test Case:
{test_case}

Original Code:
```python
{original_code}
```

Overhead Analysis:
The total memory usage during the code execution is: {total_memory_usage} MB*s.
The total execution time is: {total_execution_time} s.
The maximum memory peak requirement is: {max_memory_usage} MB.
# The profiler results are: 
# {reports}

Optimization Rules:
- Encapsulate the optimized code within a Python code block (i.e., ```python\n[Your Code Here]\n```).
- Do not include the test case within the code block.
- Focus solely on code optimization; test cases are already provided.
- Ensure the provided test case passes with your optimized solution.
'''


class EffiLearner:
    def __init__(
        self,
        corrects: Programs,
        description: str,
        log_path: str = "logs/temp.log",
    ):
        self.corrects = corrects
        self.description = description

        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("EffiLearner")
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
            h.close()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(fh)
        self.logger.propagate = False
    
    async def _run_single(self, correct: Program, generations:int=5):
        result = {}
        solutions = []
        
        self.logger.info(f"Original: {correct.id}\n{correct.code}\n")
        for gen in tqdm(range(1, generations + 1), desc="Generation", position=1, leave=False):
            result.setdefault(gen, solutions.copy())
            self.logger.info(f"=== Generation {gen} ===")
            results = Tester.run(correct, profiling=True)
            patch = await Models.run(
                system=SYSTEM,
                user=USER.format(
                    description=self.description,
                    test_case=str(Tester.testcases),
                    original_code=correct.code,
                    total_memory_usage=results.mem_usage(),
                    total_execution_time=results.exec_time(),
                    max_memory_usage=results.mem_usage_max(),
                    reports=results.report()
                ))
            if patch is None: continue
            patch = Program(
                id=correct.id,
                code=patch,
                ext=correct.ext,
            )
            results = Tester.run(patch)
            passed = Tester.is_all_pass(results)
            self.logger.info(
                f"Patch: {Status.PASSED if passed else Status.FAILED}\n{patch}\n")
            if passed:
                solutions.append(patch)
                for remaining in range(gen, generations + 1):
                    result.setdefault(remaining, solutions.copy())
                break
        return result
    
    async def run(self, generations:int=5):
        results = {}
        for correct in tqdm(self.corrects, desc="Correct", position=0):
            results[correct.id] = await self._run_single(correct, generations)
        return results
