import ast
from tqdm import tqdm

from ..utils import Randoms
from ..genetic import Selection, Variation, Fitness
from ..execution import Program, Programs, Tester

class MooRepair:
    def __init__(
        self,
        buggys: Programs,
        references: Programs,
        assignment: dict,
        ablation: str = None,
    ):
        self.buggys = buggys
        self.references = references
        self.variation = Variation(assignment, ablation)
        self.ablation = ablation
        self._patch_uid = 0

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
        references = Randoms.sample(self.references, pop_size)
        candidates = self.variation.correct(buggy, references)
        while len(candidates) < pop_size:
            candidates.append(buggy.copy())
        for patch in candidates:
            if not self._syntax_check(patch):
                patch = buggy.copy()
            self._assign_patch_id(patch)
            population.append(patch)
        return population

    def _termination(self, solutions: list[Program]) -> bool:
        early_stop = False
        if self.ablation == "no_early_stop":
            return early_stop
        for patch in solutions:
            if patch.fitness["f_fail"] == 0:
                early_stop = True
                break
        return early_stop

    def _run_single(self, buggy: Program, generations: int, pop_size: int) -> dict:
        result = {}
        solutions = []
        fitness = Fitness(buggy)
        
        self.selection = Selection(fitness, ablation=self.ablation)
        
        # Initialization
        population = self._init_population(buggy, pop_size)
        for pop in population:
            fitness.evaluate(pop)
            if not Tester.is_all_pass(pop.results): continue
            solutions.append(pop)

        for gen in tqdm(range(1, generations + 1), desc="Generation", position=1, leave=False):
            # Early termination
            if self._termination(solutions):
                for remaining in range(gen, generations + 1):
                    result.setdefault(remaining, solutions.copy())
                break

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
                    fitness.evaluate(child)
                    survivors.append(child)
                else: continue

                if not Tester.is_all_pass(child.results): continue
                solutions.append(child)

            # Prepare next generation
            population = survivors

            result.setdefault(gen, solutions.copy())
        return result

    def run(self, generations: int = 4, pop_size: int = 6) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._run_single(buggy, generations, pop_size)
        return results
