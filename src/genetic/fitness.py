import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
from pymoo.indicators.hv import HV

from ..utils import TED
from ..execution import Tester, Program, Programs


class Fitness:
    '''
    # Objective Functions
    The objective functions are defined as follows:
    F1: Similarity - Line-level Edit Distance
        두 프로그램의 라인 기반 편집 거리 (낮을수록 좋음)
        Range: [0, inf]
    F2: Similarity - AST-level Edit Distance
        두 프로그램의 AST 기반 편집 거리 (낮을수록 좋음)
        Range: [0, inf]
    F3: Efficiency - Execution Time
        두 프로그램의 실행 시간 유사도
        Range: [0, inf]
    F4: Efficiency - Memory Usage
        두 프로그램의 메모리 사용량 유사도
        Range: [0, inf]
    '''
    OBJECTIVES = ['f1', 'f2', 'f3', 'f4']
    GUIDELINES = {
        "f1": "  - Minimize the line-level edit distance",
        "f2": "  - Minimize the AST-level edit distance",
        "f3": "  - Minimize the execution time",
        "f4": "  - Minimize the memory usage",
    }

    def __init__(self, objectives:list=OBJECTIVES):
        self.OBJ_FUNC_MAP = {
            self.OBJECTIVES[0]: self.f1,
            self.OBJECTIVES[1]: self.f2,
            self.OBJECTIVES[2]: self.f3,
            self.OBJECTIVES[3]: self.f4,
        }
        self.objectives = objectives
        self.guidelines = {obj: self.GUIDELINES[obj] for obj in objectives}
    
    # [F1] Similarity - Line-level Edit Distance
    def f1(self, buggy:Program, references:Programs) -> dict:
        results = {}
        ted = TED(buggy.ext)
        for refer in tqdm(references, desc="F1", leave=False):
            results[refer.id] = ted.compute_levenshtein_led(
                buggy.code, refer.code)
        return results
    
    # [F2] Similarity - AST-level Edit Distance
    def f2(self, buggy:Program, references:Programs) -> dict:
        results = {}
        ted = TED(buggy.ext)
        for refer in tqdm(references, desc="F2", leave=False):
            results[refer.id] = ted.compute_levenshtein_ted(
                buggy.code, refer.code)
        return results

    # [F3] Efficiency - Execution Time
    def f3(self, buggy:Program, references:Programs) -> dict:
        results = {}
        for refer in tqdm(references, desc="F3", leave=False):
            refer_tests = Tester.run(refer)
            results[refer.id] = refer_tests.exec_time()
        return results
    
    # [F4] Efficiency - Memory Usage
    def f4(self, buggy:Program, references:Programs) -> dict:
        results = {}
        for refer in tqdm(references, desc="F4", leave=False):
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
            F_max = np.array([[s for s in scores.values()]], dtype=float)
        elif isinstance(scores, list):
            F_max = np.array([[s for s in scores]], dtype=float)
        else:
            raise ValueError("Scores must be a dict or a list")
        F_min = -F_max
        # set a reference point slightly worse than the worst point(0.0)
        ref = np.full(F_min.shape[1], 0.1)
        hv = HV(ref_point=ref)
        return hv.do(F_min)
    