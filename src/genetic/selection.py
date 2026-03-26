import statistics

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population

from .fitness import Fitness
from ..execution import Program, TestCase, Status, Tester
from ..utils import ETC, Randoms

class Selection:
    """EvoFix three-step selection.

    Step 1 – survivor_selection: NSGA-II on (f_fail, f_time, f_mem)
    Step 2 – assign_strategies:  SUS based on per-objective improvement rates
    Step 3 – build_pairs:        complementarity-based rank sampling → (p1, p2, t*)
    """

    STRATEGIES = ["f_fail", "f_time", "f_mem"]

    @classmethod
    def delta(cls, before: float, after: float) -> float:
        """Improvement rate ∈ [-1, 1]; 0 when denominator is zero."""
        denom = before + after
        if denom == 0.0:
            return 0.0
        return (before - after) / denom

    # ------------------------------------------------------------------ #
    # Step 1: Survivor Selection (NSGA-II)                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def survivor_selection(cls, population: list[Program], pop_size: int) -> list[Program]:
        """Keep *pop_size* individuals using NSGA-II Pareto ranking + crowding distance."""
        if len(population) <= pop_size:
            return population

        keys = [p.id for p in population]
        F = np.array(
            [
                [
                    p.fitness["f_fail"],
                    p.fitness["f_time"],
                    p.fitness["f_mem"],
                ]
                for p in population
            ],
            dtype=float,
        )

        X = np.zeros((len(keys), 1))
        pop_pymoo = Population.new("X", X, "F", F)
        pop_pymoo.set("key", np.array(keys, dtype=object))
        problem = Problem(n_var=1, n_obj=3, xl=np.array([0.0]), xu=np.array([1.0]))

        algo = NSGA2(pop_size=pop_size)
        n_survive = min(pop_size, len(pop_pymoo))
        survivors = algo.survival.do(problem, pop_pymoo, n_survive=n_survive)
        selected_ids = set(survivors.get("key").tolist())

        return [p for p in population if p.id in selected_ids]

    # ------------------------------------------------------------------ #
    # Step 2: Repair Strategy Selection via SUS                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def repair_strategy(cls, p: Program):
        """Assign p.strategy to each individual using SUS on improvement rates."""
        if p.prev_fitness is None:
            # First generation: uniform weights
            weights = [1.0, 1.0, 1.0]
        else:
            pf = p.prev_fitness
            cf = p.fitness

            def safe(key: str) -> float:
                b = pf[key]
                a = cf[key]
                return cls.delta(b, a)

            delta_fail = safe("f_fail")
            delta_time = safe("f_time")
            delta_mem  = safe("f_mem")
            weights = [
                max(delta_fail + 1.0, 0.0),
                max(delta_time + 1.0, 0.0),
                max(delta_mem  + 1.0, 0.0),
            ]

        total = sum(weights)
        if total == 0.0:
            weights = [1.0, 1.0, 1.0]
            total = 3.0

        # SUS: single pointer
        pointer = Randoms.uniform(0, total)
        cumulative = 0.0
        chosen = cls.STRATEGIES[0]
        for strategy, w in zip(cls.STRATEGIES, weights):
            cumulative += w
            if pointer <= cumulative:
                chosen = strategy
                break
        p.strategy = chosen

    # ------------------------------------------------------------------ #
    # Step 3: Parent Selection via Complementarity                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def _compute_thresholds(cls, population: list[Program]) -> tuple[float, float]:
        """θ_time and θ_mem as population-wide median (per test case)."""
        times, mems = [], []
        for p in population:
            if p.results is None:
                Tester.run(p, profiling=True)
            for tr in p.results:
                if tr.result:
                    times.append(tr.result.runtime)
                    mems.append(tr.result.memory)
        theta_time = statistics.median(times) if times else 0.0
        theta_mem  = statistics.median(mems)  if mems  else 0.0
        return theta_time, theta_mem

    @classmethod    
    def _weakness_set(cls,
       p: Program, strategy: str, theta_time: float, theta_mem: float
    ) -> set:
        """S1: test cases where p is weak according to strategy."""
        if p.results is None:
            Tester.run(p, profiling=True)
        s1 = set()
        for tr in p.results:
            if tr.result is None:
                continue
            tc = tr.testcase
            if strategy == "f_fail" and tr.result.status != Status.PASSED:
                s1.add(tc.id)
            elif strategy == "f_time" and tr.result.runtime > theta_time:
                s1.add(tc.id)
            elif strategy == "f_mem" and tr.result.memory > theta_mem:
                s1.add(tc.id)
        return s1

    @classmethod
    def _strength_set(
        cls, p: Program, strategy: str, theta_time: float, theta_mem: float
    ) -> set:
        """S2: test cases where p is strong according to strategy."""
        if p.results is None:
            Tester.run(p, profiling=True)
        s2 = set()
        for tr in p.results:
            if tr.result is None:
                continue
            tc = tr.testcase
            if strategy == "f_fail" and tr.result.status == Status.PASSED:
                s2.add(tc.id)
            elif strategy == "f_time" and tr.result.runtime <= theta_time:
                s2.add(tc.id)
            elif strategy == "f_mem" and tr.result.memory <= theta_mem:
                s2.add(tc.id)
        return s2

    @classmethod
    def _complementarity(
        cls,
        p1: Program,
        p2: Program,
        strategy: str,
        theta_time: float,
        theta_mem: float,
    ) -> float:
        """Fraction of p1's weakness test cases that p2 handles well."""
        s1 = cls._weakness_set(p1, strategy, theta_time, theta_mem)
        if not s1:
            return 0.0
        s2 = cls._strength_set(p2, strategy, theta_time, theta_mem)
        return len(s1 & s2) / len(s1)

    @classmethod
    def _representative_testcase(
        cls,
        p1: Program,
        p2: Program,
        strategy: str,
        theta_time: float,
        theta_mem: float,
    ) -> TestCase | None:
        """Select t* from S1 ∩ S2; None if intersection is empty."""
        s1 = cls._weakness_set(p1, strategy, theta_time, theta_mem)
        s2 = cls._strength_set(p2, strategy, theta_time, theta_mem)
        overlap_ids = s1 & s2
        if not overlap_ids:
            return Randoms.choice(Tester.testcases)

        # Build lookup for p1 test results by tc id
        p1_by_id = {tr.testcase.id: tr for tr in p1.results if tr.result}
        p2_by_id = {tr.testcase.id: tr for tr in p2.results if tr.result}

        if strategy == "f_fail":
            tc_id = Randoms.choice(list(overlap_ids))
            return p1_by_id[tc_id].testcase

        # For f_time / f_mem pick the test case with the largest difference
        best_id = max(
            overlap_ids,
            key=lambda tid: (
                (p1_by_id[tid].result.runtime - p2_by_id[tid].result.runtime)
                if strategy == "f_time"
                else (p1_by_id[tid].result.memory - p2_by_id[tid].result.memory)
            ),
        )
        return p1_by_id[best_id].testcase
    
    @classmethod
    def _get_pair(cls, p1: Program, candidates: list[Program], strategy: str, theta_time: float, theta_mem: float, n: int) -> Program:
        # Score each candidate by complementarity
        scores = [
            cls._complementarity(p1, p2, strategy, theta_time, theta_mem)
            for p2 in candidates
        ]

        # Rank-based weights (rank 1 = highest complementarity)
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        weights = [0.0] * len(candidates)
        for rank, idx in enumerate(order):
            weights[idx] = n - rank  # rank 1 → weight n

        total_w = sum(weights)
        if total_w == 0.0:
            return None
        else:
            r = Randoms.uniform(0, total_w)
            cumulative = 0.0
            p2 = candidates[-1]
            for p, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    p2 = p
                    break
        return p2

    @classmethod
    def build_pairs(
        cls, population: list[Program]
    ) -> list[tuple[Program, Program, TestCase | None]]:
        """Build (p1, p2, t*) pairs using complementarity rank sampling."""
        theta_time, theta_mem = cls._compute_thresholds(population)
        pairs = []
        pop_size = len(population)

        Randoms.shuffle(population)  # Randomize order to avoid bias
        for p1 in population:
            strategy = p1.strategy or "f_fail"
            candidates = [p for p in population if p.id != p1.id]
            if not candidates: continue
            p2 = cls._get_pair(p1, candidates, strategy, theta_time, theta_mem, pop_size)
            if not p2: continue
            t_star = cls._representative_testcase(p1, p2, strategy, theta_time, theta_mem)
            pairs.append((p1, p2, t_star))
            # Limit number of pairs to half the population size
            if len(pairs) >= pop_size // 2: break
        return pairs
    
    @classmethod
    def run(cls, population: list[Program], pop_size: int, selection: bool) -> list[tuple[Program, Program, TestCase | None]]:
        """Run the full selection process and return (p1, p2, t*) pairs."""
        fitnesses = [Fitness.evaluate(p) for p in population]
        if selection: # Random selection
            population = Randoms.sample(population, pop_size)
            pairs = []
            for p1 in population:
                p1.strategy = Randoms.choice(cls.STRATEGIES)
                candidates = [p for p in population if p.id != p1.id]
                if not candidates: continue
                candidates.append(None)
                p2 = Randoms.choice(candidates)
                if p2 is None: continue
                t_star = Randoms.choice(Tester.testcases)
                pairs.append((p1, p2, t_star))
                # Limit number of pairs to half the population size
                if len(pairs) >= pop_size // 2: break
            return pairs
        
        population = cls.survivor_selection(population, pop_size)
        for p in population:
            cls.repair_strategy(p)
        return cls.build_pairs(population)
    
    # ---------------------------------------------------------------- #
    # Reference selection                                              #
    # ---------------------------------------------------------------- #
    
    @classmethod
    def one(cls, buggy: Program, references: list[Program], selection: bool) -> Program:
        """Select a single reference program from the provided list."""
        if selection: # Random selection
            return Randoms.choice(references)
        cls.repair_strategy(buggy)
        theta_time, theta_mem = cls._compute_thresholds(references)
        p2 = cls._get_pair(buggy, references, buggy.strategy, theta_time, theta_mem, len(references)+1)
        return p2

    # ---------------------------------------------------------------- #
    # Final solution selection                                         #
    # ---------------------------------------------------------------- #

    @classmethod
    def prioritization(cls, population: list[Program]) -> Program | None:
        """Pick the program with the smallest mean of min-max normalized (f_time, f_mem).
        Assumes all programs in population have already passed all test cases."""
        if not population:
            return None
        if len(population) == 1:
            return population[0]

        fitnesses = [Fitness.evaluate(p) for p in population]
        time_vals = [p.fitness["f_time"] for p in population]
        mem_vals  = [p.fitness["f_mem"]  for p in population]

        def _normalize(vals: list[float]) -> list[float]:
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return [0.0] * len(vals)
            return [(v - lo) / (hi - lo) for v in vals]

        time_n = _normalize(time_vals)
        mem_n  = _normalize(mem_vals)

        scores = [ETC.divide(time_n[i] + mem_n[i], 2.0) for i in range(len(population))]
        return population[int(np.argmin(scores))]
