import ast
import logging
from pathlib import Path
from tqdm import tqdm

from ..genetic import Selection, Variation, Fitness
from ..execution import Program, Programs, Tester


class MooRepair:
    def __init__(
        self,
        buggys: Programs,
        references: Programs,
        assignment: dict,
        selection: bool,
        log_path: str = "logs/temp.log",
    ):
        self.buggys = buggys
        self.references = references
        self.variation = Variation(assignment)
        self.selection = Selection(selection)
        self._patch_uid = 0

        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("MooRepair")
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
            h.close()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(fh)
        self.logger.propagate = False

    def _assign_patch_id(self, patch: Program) -> None:
        self._patch_uid += 1
        patch.id = f"pop_{self._patch_uid}"

    def _syntax_check(self, program: Program) -> bool:
        try:
            ast.parse(program.code)
            return True
        except Exception: pass
        return False

    def _init_population(self, buggy: Program, pop_size: int) -> list[Program]:
        population = []
        tbar = tqdm(total=pop_size, desc="Population", position=1, leave=False)
        while len(population) < pop_size:
            references = []
            needed = pop_size - len(population)
            for _ in range(needed):
                references.append(self.selection.one(buggy, self.references))
            candidates = self.variation.correct(buggy, references)
            for patch in candidates:
                if self._syntax_check(patch):
                    self._assign_patch_id(patch)
                    population.append(patch)
                    tbar.update(1)
        tbar.close()
        return population

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
            
            delta_fail = self.selection.delta(b_fail, s_fail)
            delta_time = self.selection.delta(b_time, s_time)
            delta_mem  = self.selection.delta(b_mem,  s_mem)

            if delta_fail == 1.0 and delta_time > 0 and delta_mem > 0:
                log += f" >>> Early Stopped!"
                early_stop = True
            log += f"\n{s.code}\n"
            self.logger.info(log)
        return early_stop

    def _run_single(self, buggy: Program, generations: int, pop_size: int) -> dict:
        result = {}
        solutions = []
        buggy_fitness = Fitness.evaluate(buggy)
        # Initialization
        population = self._init_population(buggy, pop_size)
        for pop in population:
            results = Tester.run(pop)
            if not Tester.is_all_pass(results): continue
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
            survivors = self.selection.survivor_selection(population, pop_size)
            self.selection.repair_strategy(survivors)
            pairs = self.selection.parent_pairs(survivors)

            # Variation
            offspring = self.variation.run(pairs)

            # Validation
            for child in offspring:
                if self._syntax_check(child):
                    self._assign_patch_id(child)
                    survivors.append(child)
                else: continue
                
                results = Tester.run(child)
                if not Tester.is_all_pass(results): continue
                solutions.append(child)
            
            # Prepare next generation
            population = survivors

        return result

    def run(self, generations: int = 4, pop_size: int = 6) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._run_single(buggy, generations, pop_size)
        return results
