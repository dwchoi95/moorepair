import os
import ast
import csv
import time
from tqdm import tqdm

from ..genetic import Selection, Variation, Fitness
from ..execution import Program, Programs, Tester


INIT_STATS_PATH = "init_stats.csv"
INIT_STATS_COLS = ["ProblemID", "BuggyID", "LLM", "Loop",
                   "Requested", "Generated", "SyntaxPassed"]


class MooRepair:
    # Each ablation disables/replaces exactly one component
    # (except `random`, which replaces ALL selection steps):
    #   random         – survivor/strategy/pairing all -> random (RQ2 baseline)
    #   no_crossover   – variation by mutation only (budget-matched)
    #   no_mutation    – variation by crossover only (budget-matched)
    #   rand_survivor  – NSGA-II survivor selection -> random sampling
    #   rand_strategy  – SUS strategy assignment    -> random strategy
    #   rand_pairing   – complementarity pairing    -> random pairing
    #   no_early_stop  – disable early termination criterion
    ABLATIONS = {
        "random", "no_crossover", "no_mutation",
        "rand_survivor", "rand_strategy", "rand_pairing", "no_early_stop",
    }

    def __init__(
        self,
        buggys: Programs,
        references: Programs,
        assignment: dict,
        ablation: str = None,
    ):
        assert ablation is None or ablation in self.ABLATIONS, \
            f"Unknown ablation: {ablation}"
        self.buggys = buggys
        self.references = references
        self.assignment_id = str(assignment.get("id", ""))
        self.variation = Variation(assignment)
        self.ablation = ablation
        self.selection = Selection(
            rand=(ablation == "random"),
            rand_survivor=(ablation == "rand_survivor"),
            rand_strategy=(ablation == "rand_strategy"),
            rand_pairing=(ablation == "rand_pairing"),
        )
        self.times = {}  # buggy_id -> {gen: elapsed seconds}
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

    def _log_init_stats(self, buggy: Program, loop: int,
                        requested: int, generated: int, passed: int) -> None:
        from ..llms import Models
        llm = getattr(Models, "model", "")
        exists = os.path.exists(INIT_STATS_PATH) and \
            os.path.getsize(INIT_STATS_PATH) > 0
        with open(INIT_STATS_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(INIT_STATS_COLS)
            writer.writerow([self.assignment_id, buggy.id, llm, loop,
                             requested, generated, passed])

    def _init_population(self, buggy: Program, pop_size: int) -> list[Program]:
        population = []
        references = []
        for _ in range(pop_size):
            references.append(self.selection.one(buggy, self.references))
        candidates = self.variation.correct(buggy, references)
        passed = 0
        for patch in candidates:
            if self._syntax_check(patch):
                self._assign_patch_id(patch)
                population.append(patch)
                passed += 1
        self._log_init_stats(buggy, 1, pop_size, len(candidates), passed)
        return population

    def _termination(self, solutions: list[Program], b_fitness: dict) -> bool:
        early_stop = False
        b_fail = b_fitness["f_fail"]
        b_time = b_fitness["f_time"]
        b_mem  = b_fitness["f_mem"]

        for s in solutions:
            s_fitness = Fitness.evaluate(s)
            s_fail = s_fitness["f_fail"]
            s_time = s_fitness["f_time"]
            s_mem  = s_fitness["f_mem"]

            delta_fail = self.selection.delta(b_fail, s_fail)
            delta_time = self.selection.delta(b_time, s_time)
            delta_mem  = self.selection.delta(b_mem,  s_mem)

            if delta_fail == 1.0 and delta_time > 0 and delta_mem > 0:
                early_stop = True
        return early_stop

    def _run_single(self, buggy: Program, generations: int, pop_size: int) -> dict:
        result = {}
        times = {}
        solutions = []
        start = time.perf_counter()
        buggy_fitness = Fitness.evaluate(buggy)
        # Initialization
        population = self._init_population(buggy, pop_size)
        for pop in population:
            results = Tester.run(pop)
            if not Tester.is_all_pass(results): continue
            solutions.append(pop)

        # result[gen] holds the solutions found AFTER generation gen's
        # variation, so every LLM call contributes to a reported snapshot
        for gen in tqdm(range(1, generations + 1), desc="Generation", position=1, leave=False):
            # Early termination: skip remaining variation rounds (saves LLM calls)
            if self.ablation != "no_early_stop" and \
                    self._termination(solutions, buggy_fitness):
                elapsed = time.perf_counter() - start
                for remaining in range(gen, generations + 1):
                    result.setdefault(remaining, solutions.copy())
                    times.setdefault(remaining, elapsed)
                break

            # Selection
            survivors = self.selection.survivor_selection(population, pop_size)
            self.selection.repair_strategy(survivors)
            pairs = self.selection.parent_pairs(survivors)

            # Variation
            offspring = self.variation.run(
                pairs,
                crossover=(self.ablation != "no_crossover"),
                mutation=(self.ablation != "no_mutation"),
            )

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

            result.setdefault(gen, solutions.copy())
            times.setdefault(gen, time.perf_counter() - start)

        self.times[buggy.id] = times
        return result

    def run(self, generations: int = 4, pop_size: int = 6) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._run_single(buggy, generations, pop_size)
        return results
