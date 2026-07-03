from functools import cache
from ..utils.etc import ETC
from ..execution.program import Program
from ..execution.tester import Tester

class Fitness:
    """EvoFix fitness evaluator: 4 objectives (f_fail, f_ted, f_time, f_mem).

    f_fail = |failed tests| / |T|
    f_ted  = Levenshtein edit distance of ASTs
    f_time = max(exec_time per test) (sec)
    f_mem  = max(mem_usage per test) (MB)
    """
    def __init__(self, buggy: Program):
        self.evaluate.cache_clear()
        self.buggy = buggy
        self.evaluate(buggy)

    @cache
    def evaluate(self, program: Program) -> dict:
        results = Tester.run(program)
        total = len(results)
        _, failed = Tester.tests_split(results)
        f_fail = len(failed) / total
        f_ted = ETC.ted(self.buggy.code, program.code)

        fitness = {"f_fail": f_fail, "f_ted": f_ted,
                   "f_time": results.exec_time_max(), "f_mem": results.mem_usage_max()}

        program.fitness = fitness
        return fitness
