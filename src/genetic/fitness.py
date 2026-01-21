import logging
import warnings
warnings.filterwarnings("ignore")

import lizard
import numpy as np
from tqdm import tqdm
from pymoo.indicators.hv import HV
from functools import cache
from codebleu import calc_codebleu

from ..utils import ETC
from ..execution import Tester, Program, Programs

class Fitness:
    '''
    # Objective Functions
    The objective functions are defined as follows:
    F1: Accuracy
        두 프로그램의 passed test case set의 교집합
        Range: [0, 1]
    F2: Similarity
        두 프로그램의 CodeBLEU 유사도
        Range: [0, 1]
    F3: Efficiency
        두 프로그램의 Cyclomatic Complexity 유사도
        Range: [0, 1]
    '''
    OBJECTIVES = ['f1', 'f2', 'f3', 'f4', 'f5']
    
    def __init__(self, objectives:list=OBJECTIVES):
        self.OBJ_FUNC_MAP = {
            self.OBJECTIVES[0]: self.accuracy,
            self.OBJECTIVES[1]: self.similarity,
            self.OBJECTIVES[2]: self.efficiency,
            self.OBJECTIVES[3]: self.runtime,
            self.OBJECTIVES[4]: self.memory,
        }
        self.objectives = objectives
        
        self.guidelines = {
            "f1": "  - Fix the failed test cases of buggy program to pass all tests",
            "f2": "  - Minimize the changes of a buggy program while fixing bugs",
            "f3": "  - Minimize the cyclomatic complexity of a buggy program",
            "f4": "  - Minimize the runtime of a buggy program",
            "f5": "  - Minimize the memory usage of a buggy program",
        }
    
    
    # [F1] Accuracy
    def accuracy(self, buggy:Program, references:Programs) -> dict:
        results = {}
        Tester.run(buggy)
        for refer in tqdm(references, desc="F1", leave=False):
            Tester.run(refer)
            results[refer.id] = self.__calc_tests(buggy, refer)
        return results
    
    def __calc_tests(self, buggy:Program, refer:Program) -> float:
        # Get passed test cases
        b_passed, b_failed = Tester.tests_split(buggy)
        r_passed, r_failed = Tester.tests_split(refer)
        
        pass2pass = len(set(b_passed) & set(r_passed))
        fail2pass = len(set(b_failed) & set(r_passed))
        return ETC.divide(pass2pass + fail2pass, len(Tester.testcases))
    
    
    # [F2] Similarity
    def similarity(self, buggy:Program, references:Programs) -> dict:
        results = {}
        for refer in tqdm(references, desc="F2", leave=False):
            results[refer.id] = self.__codebleu(
                buggy.code, refer.code, buggy.ext)
        return results
    
    def __codebleu(self, code1:str, code2:str, language:str="c") -> float:
        language = language.lower()
        if language == "c++":
            language = "cpp"
        if code2 is None or code2.strip() == "":
            return 0.0
        logging.disable(logging.WARNING)
        try:
            return calc_codebleu(
                [code1],
                [code2],
                lang=language
            )["codebleu"]
        finally:
            logging.disable(logging.NOTSET)

    # [F3] Efficiency
    def efficiency(self, buggy:Program, references:Programs) -> dict:
        results = {}
        # b_cyclomatic = self.__cyclomatic(buggy)
        for refer in tqdm(references, desc="F3", leave=False):
            results[refer.id] = self.__cyclomatic(refer)
        return results
    
    def __cyclomatic(self, program:Program) -> int:
        analysis = lizard.analyze_file.analyze_source_code(
            f"file.{program.ext}", program.code)
        return analysis.average_cyclomatic_complexity
    
    # [F4] Runtime
    def runtime(self, buggy:Program, references:Programs) -> dict:
        results = {}
        # Tester.run(buggy)
        # b_runtime = buggy.results.runtime()
        for refer in tqdm(references, desc="F4", leave=False):
            Tester.run(refer)
            results[refer.id] = refer.results.runtime()
        return results
    
    # [F5] Memory
    def memory(self, buggy:Program, references:Programs) -> dict:
        results = {}
        # Tester.run(buggy)
        # b_memory = buggy.results.memory()
        for refer in tqdm(references, desc="F5", leave=False):
            Tester.run(refer)
            results[refer.id] = refer.results.memory()
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
    