from ..execution.program import Program
from ..execution.tester import Tester

class Fitness:
    """EvoFix fitness evaluator: 3 objectives (f_fail, f_time, f_mem).

    f_fail = |failed tests| / |T|           minimise, ∈ [0.0, 1.0]
    f_time = max(exec_time per test) (sec)  minimise; ∞ when f_fail > 0
    f_mem  = max(mem_usage per test) (MB)   minimise; ∞ when f_fail > 0

    Aggregation is MAX (OJ grades on worst-case test).
    f_time and f_mem are set to infinity for incorrect programs so that
    they are automatically dominated in Pareto ranking.
    """

    @staticmethod
    def evaluate(program: Program) -> dict:
        results = Tester.run(program)
        total = len(results)
        _, failed = Tester.tests_split(results)
        f_fail = len(failed) / total

        if f_fail == 0.0:
            f_time = max(
                (tr.result.runtime for tr in results if tr.result),
                default=0.0,
            )
            f_mem = max(
                (tr.result.memory for tr in results if tr.result),
                default=0.0,
            )
        else:
            f_time = float("inf")
            f_mem = float("inf")

        fitness = {"f_fail": f_fail, "f_time": f_time, "f_mem": f_mem}

        program.fitness = fitness
        return fitness
