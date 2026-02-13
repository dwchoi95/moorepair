import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from ..utils import DIST, ETC
from ..execution import Tester, Program


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
    F3: Similarity - Code Coverage Distance
        두 프로그램의 코드 커버리지 거리
        Range: [0, 1]
    F4: Similarity - Tree Edit Distance
        두 프로그램의 트리 편집 거리 
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
        self.strengths = {
            "f1": "  - Passes the test cases that the Student Program passed.",
            "f2": "  - Passes the test cases that the Student Program failed.",           
            "f3": "  - Similar code coverage to the Student Program.",
            "f4": "  - Low tree edit distance from the Student Program.",            
            "f5": "  - Short execution time.",            
            "f6": "  - Low memory usage.",
        }

    # [F1] Correctness - Passed2Passed Test Cases
    def f1(self, buggy:Program, references:list[Program]) -> dict:
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
    def f2(self, buggy:Program, references:list[Program]) -> dict:
        results = {}
        b_results = Tester.run(buggy)
        b_passed, b_failed = Tester.tests_split(b_results)
        for refer in tqdm(references, desc="F2", leave=False):
            r_results = Tester.run(refer)
            r_passed, r_failed = Tester.tests_split(r_results)
            intersect = b_failed.intersection(r_passed)
            results[refer.id] = 1 - ETC.divide(len(intersect), len(b_results))
        return results
    
    # [F3] Similarity - Code Coverage Distance
    def f3(self, buggy:Program, references:list[Program]) -> dict:
        results = {}
        b_results = Tester.run(buggy)
        b_coverage = b_results.get_coverage_line()

        for refer in tqdm(references, desc="F3", leave=False):
            r_results = Tester.run(refer)
            r_coverage = r_results.get_coverage_line()
            results[refer.id] = DIST.compute_ccd(
                b_coverage, r_coverage)
        return results
    
    # [F4] Similarity - Tree Edit Distance
    def f4(self, buggy:Program, references:list[Program]) -> dict:
        results = {}
        DIST.set_language(buggy.ext)
        for refer in tqdm(references, desc="F4", leave=False):
            results[refer.id] = DIST.compute_levenshtein_ted(
                buggy.code, refer.code)
        return results

    # [F5] Efficiency - Execution Time
    def f5(self, buggy:Program, references:list[Program]) -> dict:
        results = {}
        for refer in tqdm(references, desc="F5", leave=False):
            refer_tests = Tester.run(refer)
            results[refer.id] = refer_tests.exec_time()
        return results
    
    # [F6] Efficiency - Memory Usage
    def f6(self, buggy:Program, references:list[Program]) -> dict:
        results = {}
        for refer in tqdm(references, desc="F6", leave=False):
            refer_tests = Tester.run(refer)
            results[refer.id] = refer_tests.mem_usage()
        return results
    
    def evaluate(self, buggy:Program, references:list[Program]) -> dict[str, dict[str, float]]:
        scores = {obj: self.OBJ_FUNC_MAP[obj](buggy, references) 
                   for obj in self.objectives}
        X = np.array([[scores[o][r.id] for o in self.objectives] for r in references], dtype=float)
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        Xn = scaler.fit_transform(X)
        normalized = {
            r.id: {o: float(Xn[i, j]) for j, o in enumerate(self.objectives)}
            for i, r in enumerate(references)}
        return normalized
    
    def run(self, buggy:Program, references:list[Program]) -> dict[str, list[float]]:
        normalized = self.evaluate(buggy, references)
        scores = {refer.id: [normalized[refer.id][obj] for obj in self.objectives]
                    for refer in references}
        return scores
    
