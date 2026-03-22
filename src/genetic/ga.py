import ast
import logging
from pathlib import Path
from tqdm import tqdm

from .selection import Selection
from .variation import Variation
from .fitness import Fitness
from ..execution import Program, Programs, Tester
from ..utils import ETC


class GeneticAlgorithm:
    def __init__(
        self,
        buggys: Programs,
        description: str,
        log_path: str = "logs/temp.log",
    ):
        self.buggys = buggys
        self.variation = Variation(description)
        self._patch_uid = 0

        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("GA")
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
            h.close()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(fh)
        self.logger.propagate = False

    # ---------------------------------------------------------------- #
    # Helpers                                                          #
    # ---------------------------------------------------------------- #

    def _assign_patch_id(self, patch: Program) -> None:
        self._patch_uid += 1
        patch.id = f"pop_{self._patch_uid}"

    def _is_uniq(self, new: Program, population: list[Program]) -> bool:
        new_code = ETC.normalize_code(new.code)
        return all(ETC.normalize_code(p.code) != new_code for p in population)

    @staticmethod
    def _syntax_check(program: Program) -> bool:
        if program.ext.lower() != "py":
            return False
        try:
            ast.parse(program.code)
            return True
        except SyntaxError:
            return False

    # ---------------------------------------------------------------- #
    # Initialization (syntax-check only, no test execution)            #
    # ---------------------------------------------------------------- #

    def _init_population(self, buggy: Program, pop_size: int) -> list[Program]:
        population = []
        tbar = tqdm(total=pop_size, desc="Population", position=1, leave=False)
        while len(population) < pop_size:
            needed = pop_size - len(population)
            candidates = self.variation.run_init(buggy, needed)
            for patch in candidates:
                if self._syntax_check(patch) and self._is_uniq(patch, population):
                    self._assign_patch_id(patch)
                    population.append(patch)
                    tbar.update(1)
        tbar.close()
        return population

    # ---------------------------------------------------------------- #
    # Termination condition (EvoFix §9)                                #
    # ---------------------------------------------------------------- #

    def _termination(self, solutions: list[Program], b_fitness: dict) -> bool:
        """Stop if any individual is fully correct AND faster AND lighter than buggy."""
        early_stop = False
        b_fail = b_fitness["f_fail"]
        b_time = b_fitness["f_time"]
        b_mem  = b_fitness["f_mem"]

        for i, s in enumerate(solutions):
            s_fitness = Fitness.evaluate(s)
            s_fail = s_fitness["f_fail"]
            s_time = s_fitness["f_time"]
            s_mem  = s_fitness["f_mem"]

            log = f"Patch {i}: CORR: {s_fail:.2f} | RUN: {s_time:.2f} | MEM: {s_mem:.2f}"
            
            delta_fail = Selection.delta(b_fail, s_fail)
            delta_time = Selection.delta(b_time, s_time)
            delta_mem  = Selection.delta(b_mem,  s_mem)

            if delta_fail == 1.0 and delta_time > 0 and delta_mem > 0:
                log += f" >>> Early Stopped!"
                early_stop = True
            log += f"\n{s.code}\n"
            self.logger.info(log)
        return early_stop

    # ---------------------------------------------------------------- #
    # Per-buggy GA run                                                 #
    # ---------------------------------------------------------------- #

    def _ga_run(self, buggy: Program, generations: int, pop_size: int) -> dict:
        result = {}
        solutions = []
        buggy_fitness = Fitness.evaluate(buggy)
        # Initialization
        population = self._init_population(buggy, pop_size)
        for pop in population:
            results = Tester.run(pop)
            if not Tester.is_all_pass(results): continue
            if self._is_uniq(pop, solutions):
                solutions.append(pop)

        self.logger.info(f"Buggy: {buggy.id}\n{buggy.code}\n")
        for gen in tqdm(range(1, generations + 1), desc="Generation", position=1, leave=False):
            if self._termination(solutions, buggy_fitness):
                for remaining in range(gen, generations + 1):
                    result.setdefault(remaining, solutions.copy())
                break
            result.setdefault(gen, solutions.copy())
            self.logger.info(f"=== Generation {gen} ===")

            # Selection
            pairs = Selection.run(population, pop_size)

            # Variation
            offspring = self.variation.run(buggy, pairs)

            # Validation
            for child in offspring:
                if self._syntax_check(child) and self._is_uniq(child, population):
                    self._assign_patch_id(child)
                    population.append(child)
                else: continue
                
                results = Tester.run(child)
                if not Tester.is_all_pass(results): continue
                if self._is_uniq(child, solutions):
                    solutions.append(child)

        return result

    # ---------------------------------------------------------------- #
    # Public entry point                                               #
    # ---------------------------------------------------------------- #

    def run(self, generations: int = 10, pop_size: int = 10) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._ga_run(buggy, generations, pop_size)
        return results
