import statistics

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population

from ..execution import Program, TestCase, Status, Tester
from ..utils import ETC, Randoms

class Selection:
    """EvoFix three-step selection.

    Step 1 – survivor_selection: NSGA-II on (f_fail, f_ted, f_time, f_mem)
    Step 2 – repair_strategy:    Roulette wheel selection on improvement rates → p.strategy
    Step 3 – parent_pairs:       complementarity-based rank sampling → (p1, p2, t*)
    """

    STRATEGIES = ["f_fail", "f_time", "f_mem"]

    def __init__(self, fitness=None, ablation:str=None):
        self.fitness = fitness
        self.ablation = ablation
        
    def delta(self, before: float, after: float) -> float:
        denom = before + after
        if denom == 0.0:
            return 0.0
        return (before - after) / denom

    def _improvement_rate(self, key: str, before: float, after: float) -> float:
        denom = before + after
        if denom == 0.0:
            return 0.0
        return (before - after) / denom

    def _strategy_weights(self, program: Program) -> list[float]:
        if program.prev_fitness is None or program.fitness is None:
            return [1.0] * len(self.STRATEGIES)

        prev = program.prev_fitness
        curr = program.fitness
        if any(key not in prev or key not in curr for key in self.STRATEGIES):
            return [1.0] * len(self.STRATEGIES)

        improvement_rates = [
            self._improvement_rate(key, prev[key], curr[key])
            for key in self.STRATEGIES
        ]
        return [max(1.0 - rate, 0.0) for rate in improvement_rates]

    def _roulette_strategy(self, weights: list[float]) -> str:
        total = sum(weights)
        if total <= 0.0:
            weights = [1.0] * len(self.STRATEGIES)
            total = float(len(self.STRATEGIES))

        pointer = Randoms.uniform(0.0, total)
        cumulative = 0.0
        for strategy, weight in zip(self.STRATEGIES, weights):
            cumulative += weight
            if pointer <= cumulative:
                return strategy
        return self.STRATEGIES[-1]

    # ------------------------------------------------------------------ #
    # Step 1: Survivor Selection (NSGA-II)                               #
    # ------------------------------------------------------------------ #

    def survivor_selection(self, population: list[Program], pop_size: int) -> list[Program]:
        """Keep *pop_size* individuals using NSGA-II Pareto ranking + crowding distance."""
        if len(population) <= pop_size:
            return population

        # Random selection
        if self.ablation == "random" or self.ablation == "rand_survivor":
            return Randoms.sample(population, pop_size)
            
        keys = [p.id for p in population]
        F = np.array(
            [
                [
                    p.fitness["f_fail"],
                    p.fitness["f_ted"],
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
        problem = Problem(n_var=1, n_obj=4, xl=np.array([0.0]), xu=np.array([1.0]))

        algo = NSGA2(pop_size=pop_size)
        n_survive = min(pop_size, len(pop_pymoo))
        survivors = algo.survival.do(problem, pop_pymoo, n_survive=n_survive)
        selected_ids = set(survivors.get("key").tolist())

        return [p for p in population if p.id in selected_ids]

    # ------------------------------------------------------------------ #
    # Step 2: Repair Strategy Selection via RWS                          #
    # ------------------------------------------------------------------ #

    def repair_strategy(self, survivors: list[Program]):
        """Assign p.strategy using roulette wheel selection on improvement rates."""
        for p in survivors:
            # Random strategy assignment
            if self.ablation == "random" or self.ablation == "rand_strategy":
                p.strategy = Randoms.choice(self.STRATEGIES)
                continue
            p.strategy = self._roulette_strategy(self._strategy_weights(p))

    # ------------------------------------------------------------------ #
    # Step 3: Parent Selection via Complementarity                       #
    # ------------------------------------------------------------------ #

    def _compute_thresholds(self, population: list[Program]) -> tuple[float, float]:
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
    
    def _weakness_set(self,
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

    def _strength_set(
        self, p: Program, strategy: str, theta_time: float, theta_mem: float
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

    def _complementarity(
        self,
        p1: Program,
        p2: Program,
        strategy: str,
        theta_time: float,
        theta_mem: float,
    ) -> float:
        """Fraction of p1's weakness test cases that p2 handles well."""
        s1 = self._weakness_set(p1, strategy, theta_time, theta_mem)
        if not s1:
            return 0.0
        s2 = self._strength_set(p2, strategy, theta_time, theta_mem)
        return len(s1 & s2) / len(s1)

    def _representative_testcase(
        self,
        p1: Program,
        p2: Program,
        strategy: str,
        theta_time: float,
        theta_mem: float,
    ) -> TestCase | None:
        """Select t* from S1 ∩ S2; None if intersection is empty."""
        s1 = self._weakness_set(p1, strategy, theta_time, theta_mem)
        s2 = self._strength_set(p2, strategy, theta_time, theta_mem)
        overlap_ids = s1 & s2
        if not overlap_ids:
            return Randoms.choice(Tester.testcases)

        # Build lookup for p1 test results by tc id
        p1_by_id = {tr.testcase.id: tr for tr in p1.results if tr.result}
        p2_by_id = {tr.testcase.id: tr for tr in p2.results if tr.result}

        if strategy in {"f_fail", "f_ted"}:
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
    
    def _get_pair(self, p1: Program, candidates: list[Program], strategy: str, theta_time: float, theta_mem: float, n: int) -> Program:
        # Score each candidate by complementarity
        scores = [
            self._complementarity(p1, p2, strategy, theta_time, theta_mem)
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

    def parent_pairs(
        self, survivors: list[Program]
    ) -> list[tuple[Program, Program, TestCase | None]]:
        """Build (p1, p2, t*) pairs using complementarity rank sampling."""
        pairs = []
        pop_size = len(survivors)

        if self.ablation == "random" or self.ablation == "rand_pairing": # Random pairing
            Randoms.shuffle(survivors)
            for p1 in survivors:
                candidates = [p for p in survivors if p.id != p1.id]
                if not candidates: continue
                p2 = Randoms.choice(candidates)
                t_star = Randoms.choice(Tester.testcases)
                pairs.append((p1, p2, t_star))
                # Limit number of pairs to half the population size
                if len(pairs) >= pop_size // 2: break
            return pairs

        theta_time, theta_mem = self._compute_thresholds(survivors)

        Randoms.shuffle(survivors)  # Randomize order to avoid bias
        for p1 in survivors:
            strategy = p1.strategy
            candidates = [p for p in survivors if p.id != p1.id]
            if not candidates: continue
            p2 = self._get_pair(p1, candidates, strategy, theta_time, theta_mem, pop_size)
            if not p2: continue
            t_star = self._representative_testcase(p1, p2, strategy, theta_time, theta_mem)
            pairs.append((p1, p2, t_star))
            # Limit number of pairs to half the population size
            if len(pairs) >= pop_size // 2: break
        return pairs

    # ---------------------------------------------------------------- #
    # Final solution selection                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def prioritization(population: list[Program]) -> Program | None:
        """Pick the program with the smallest mean of min-max normalized (f_time, f_mem).
        Assumes all programs in population have already passed all test cases."""
        if not population:
            return None
        if len(population) == 1:
            return population[0]

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
