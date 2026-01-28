import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
from pymoo.indicators.hv import HV

from ..utils import TED, ETC
from ..execution import Tester, Program, Programs


class Fitness:
    '''
    # Objective Functions
    The objective functions are defined as follows:
    F1: Correctness - Passed2Passed Test Cases
        두 프로그램이 통과한 테스트 케이스 교집합 비율
        Range: [0, 1]
    F2: Correctness - Failed2Passed Test Cases
        버기 프로그램이 실패한 테스트 케이스와 참조 프로그램이 성공한 테스트 케이스 교집합 비율
        Range: [0, 1]
    F3: Similarity - Line-level Edit Distance
        두 프로그램의 라인 기반 편집 거리
        Range: [0, inf]
    F4: Similarity - AST-level Edit Distance
        두 프로그램의 AST 기반 편집 거리 
        Range: [0, inf]
    F5: Efficiency - Execution Time
        두 프로그램의 실행 시간 유사도
        Range: [0, inf]
    F6: Efficiency - Memory Usage
        두 프로그램의 메모리 사용량 유사도
        Range: [0, inf]
    '''
    OBJECTIVES = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

    def __init__(self, objectives:list=OBJECTIVES):
        self.OBJ_FUNC_MAP = {
            self.OBJECTIVES[0]: self.f1,
            self.OBJECTIVES[1]: self.f2,
            self.OBJECTIVES[2]: self.f3,
            self.OBJECTIVES[3]: self.f4,
            self.OBJECTIVES[4]: self.f5,
            self.OBJECTIVES[5]: self.f6,
        }
        self.objectives = objectives
        self.guidelines = {
            "f1": "Preserve already-passing behavior; do not introduce regressions on previously passed test cases.",
            "f2": "Fix behaviors that cause failing test cases; handle edge cases implied by the description.",
            "f3": "Minimize the line-level edit distance from the Buggy Program; prefer small, localized edits.",
            "f4": "Minimize the AST-level edit distance; avoid large refactoring unless necessary for correctness.",
            "f5": "Reduce the execution time; prefer algorithmic simplifications and efficient data structures over micro-optimizations.",
            "f6": "Reduce the memory usage; avoid unnecessary allocations and large auxiliary structures; reuse data when possible.",
        }

    # [F1] Correctness - Passed2Passed Test Cases
    def f1(self, buggy:Program, references:Programs) -> dict:
        results = {}
        b_results = Tester.run(buggy)
        b_passed, b_failed = Tester.tests_split(b_results)
        for refer in tqdm(references, desc="F1", leave=False):
            r_results = Tester.run(refer)
            r_passed, r_failed = Tester.tests_split(r_results)
            intersect = b_passed.intersection(r_passed)
            results[refer.id] = 1 - ETC.divide(len(intersect), len(b_results))
        return results
    
    # [F2] Correctness - Failed2Passed Test Cases
    def f2(self, buggy:Program, references:Programs) -> dict:
        results = {}
        b_results = Tester.run(buggy)
        b_passed, b_failed = Tester.tests_split(b_results)
        for refer in tqdm(references, desc="F2", leave=False):
            r_results = Tester.run(refer)
            r_passed, r_failed = Tester.tests_split(r_results)
            intersect = b_failed.intersection(r_passed)
            results[refer.id] = 1 - ETC.divide(len(intersect), len(b_results))
        return results
    
    # [F3] Similarity - Line-level Edit Distance
    def f3(self, buggy:Program, references:Programs) -> dict:
        results = {}
        ted = TED(buggy.ext)
        for refer in tqdm(references, desc="F3", leave=False):
            results[refer.id] = ted.compute_levenshtein_led(
                buggy.code, refer.code)
        return results
    
    # [F4] Similarity - AST-level Edit Distance
    def f4(self, buggy:Program, references:Programs) -> dict:
        results = {}
        ted = TED(buggy.ext)
        for refer in tqdm(references, desc="F4", leave=False):
            results[refer.id] = ted.compute_levenshtein_ted(
                buggy.code, refer.code)
        return results

    # [F5] Efficiency - Execution Time
    def f5(self, buggy:Program, references:Programs) -> dict:
        results = {}
        for refer in tqdm(references, desc="F5", leave=False):
            refer_tests = Tester.run(refer)
            results[refer.id] = refer_tests.exec_time()
        return results
    
    # [F6] Efficiency - Memory Usage
    def f6(self, buggy:Program, references:Programs) -> dict:
        results = {}
        for refer in tqdm(references, desc="F6", leave=False):
            refer_tests = Tester.run(refer)
            results[refer.id] = refer_tests.mem_usage()
        return results
    
    
    def evaluate(self, buggy:Program, references:Programs) -> dict[str, list[float]]:
        results = {obj: self.OBJ_FUNC_MAP[obj](buggy, references) 
                   for obj in self.objectives}
        scores = {refer.id: [results[obj][refer.id] for obj in self.objectives]
                    for refer in references}
        return scores
    
    def run(self, buggy:Program, patch:Program) -> dict[str, float]:
        scores = {}
        for obj in self.objectives:
            scores[obj] = self.OBJ_FUNC_MAP[obj](buggy, Programs([patch]))[patch.id]
        return scores
    
    def hypervolume(self, scores:dict[str, float] | list[float]) -> float:
        if isinstance(scores, dict):
            objectives = list(scores.keys())
            x = np.array([scores[o] for o in objectives], dtype=float)
            ref = np.ones(len(objectives), dtype=float)
        elif isinstance(scores, list):
            x = np.array(scores, dtype=float)
            ref = np.ones(len(scores), dtype=float)
        hv = HV(ref_point=ref)
        return float(hv(x.reshape(1, -1)))
    