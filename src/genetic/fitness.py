import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
from pymoo.indicators.hv import HV
from functools import cache
from datasketch import MinHash

from ..utils import ETC, NodeParser
from ..execution import Tester
    
class Fitness:
    '''
    # Objective Functions
      * Correctness: F1, F2
      * Similarity: F3, F4
      * Efficiency: F5, F6
    
    ## The objective functions are defined as follows:
      * F1: Failed to Passed Testcases
        Generate a patch that fixes the buggy code to pass all testcases
        Range: [0, 1]
      * F2: Passed to Passed Testcases
        Generate a patch that maintains the passing testcases of the buggy
        Range: [0, 1]
      * F3: Static(Syntax) Similarity (ASTsequence)
        Generate a patch that minimizes the static changes in the code
        Range: [-1, 1]
      * F4: Dynamic(Behavior) Similarity (Variable Value Sequence)
        Generate a patch that minimizes the dynamic changes in the code
        Range: [-1, 1]
      * F5: Memory Usage (tracemalloc)
        Generate a patch that minimizes the memory useage of the code
        Range: [-1, 1]
      * F6: Execution Time (time)
        Generate a patch that minimizes the execution time of the code
        Range: [-1, 1]
    '''
    OBJECTIVES = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    
    def __init__(self, objectives:list=OBJECTIVES):
        self.OBJ_FUNC_MAP = {
            self.OBJECTIVES[0]: self.failed_to_passed,
            self.OBJECTIVES[1]: self.passed_to_passed,
            self.OBJECTIVES[2]: self.static_similarity,
            self.OBJECTIVES[3]: self.dynamic_similarity,
            self.OBJECTIVES[4]: self.mem_usage,
            self.OBJECTIVES[5]: self.exec_time,
        }
        self.OBJ_EVAL_MAP = {
            self.OBJECTIVES[0]: self.eval_fp,
            self.OBJECTIVES[1]: self.eval_pp,
            self.OBJECTIVES[2]: self.eval_static,
            self.OBJECTIVES[3]: self.eval_dynamic,
            self.OBJECTIVES[4]: self.eval_memory,
            self.OBJECTIVES[5]: self.eval_exec_time,
        }
        self.objectives = objectives
        self.guidelines = {
            "f1": "  - Fix the failed test cases",
            "f2": "  - Preserve the passed test cases",
            "f3": "  - Make minimal syntax changes",
            "f4": "  - Make minimal behavioral changes",
            "f5": "  - Reduce memory usage",
            "f6": "  - Reduce execution time"
        }
        
    # [F1] Failed to Passed Testcases
    def failed_to_passed(self, buggy:str, references:dict) -> dict:
        results = {}
        total_tests = len(Tester.testsuite)
        b_passed, b_failed = self._get_test_results(buggy)
        for r_id, r_code in tqdm(references.items(),
                                 total=len(references),
                                 desc="F1",
                                 leave=False):
            r_passed, r_failed = self._get_test_results(r_code)
            if len(b_failed):
                score = ETC.divide(len(b_failed & r_passed), len(b_failed))
            else:
                score = ETC.divide(len(r_passed), total_tests)
            results[r_id] = score
        return results
    
    def eval_fp(self, buggy:str, reference:str) -> float:
        total_tests = len(Tester.testsuite)
        b_passed, b_failed = self._get_test_results(buggy)
        r_passed, r_failed = self._get_test_results(reference)
        if len(b_failed):
            score = ETC.divide(len(b_failed & r_passed), len(b_failed))
        else:
            score = ETC.divide(len(r_passed), total_tests)
        return score
    
    
    # [F2] Passed to Passed Testcases
    def passed_to_passed(self, buggy:str, references:dict) -> dict:
        results = {}
        total_tests = len(Tester.testsuite)
        b_passed, b_failed = self._get_test_results(buggy)
        for r_id, r_code in tqdm(references.items(),
                                 total=len(references),
                                 desc="F2",
                                 leave=False):
            r_passed, r_failed = self._get_test_results(r_code)
            if len(b_passed):
                score = ETC.divide(len(b_passed & r_passed), len(b_passed))
            else:
                score = ETC.divide(len(r_passed), total_tests)
            results[r_id] = score
        return results
    
    def eval_pp(self, buggy:str, reference:str) -> float:
        total_tests = len(Tester.testsuite)
        b_passed, b_failed = self._get_test_results(buggy)
        r_passed, r_failed = self._get_test_results(reference)
        if len(b_passed):
            score = ETC.divide(len(b_passed & r_passed), len(b_passed))
        else:
            score = ETC.divide(len(r_passed), total_tests)
        return score
    
    def _get_test_results(self, code:str) -> tuple[set, set]:
        results = Tester.test(code)
        test_hist = Tester.get_test_hist(results)
        passed, failed = Tester.split_test_hist(test_hist)
        return set(passed), set(failed)


    # [F3] Static Similarity
    def static_similarity(self, buggy:str, references:dict) -> dict:
        results = {}
        b_minhash = self._ast2hash(buggy)
        for r_id, r_code in tqdm(references.items(),
                                 total=len(references),
                                 desc="F3",
                                 leave=False):
            r_minhash = self._ast2hash(r_code)
            results[r_id] = \
                self.__calc_static(b_minhash, r_minhash)
        return results
    
    def _ast2hash(self, code:str) -> str:
        np = NodeParser()
        np.run(code)
        m = MinHash(num_perm=128)
        for seq in np.ast_seq:
            seq_str = "->".join(seq)
            m.update(seq_str.encode('utf-8'))
        return m
    
    def eval_static(self, buggy:str, reference:str) -> float:
        r_minhash = self._ast2hash(reference)
        b_minhash = self._ast2hash(buggy)
        return self.__calc_static(b_minhash, r_minhash)
    
    def __calc_static(self, b_minhash:MinHash, r_minhash:MinHash) -> float:
        return b_minhash.jaccard(r_minhash)
    
    
    # [F4] Dynamic Similarity
    def dynamic_similarity(self, buggy:str, references:dict) -> dict:
        results = {}
        b_vvs = self._vvs(buggy)
        for r_id, r_code in tqdm(references.items(),
                                 total=len(references),
                                 desc="F4",
                                 leave=False):
            r_vvs = self._vvs(r_code)
            results[r_id] = \
                self.__calc_dynamic(b_vvs, r_vvs)
        return results
    
    def _vvs(self, code:str) -> dict:
        return {
            no: {var: list(data.values())
                         for var, data in result.vari_traces.items()}
            for no, result in Tester.trace(code)
        }
                        
    def eval_dynamic(self, buggy:str, reference:str) -> float:
        r_vari_traces = self._vvs(reference)
        b_vari_traces = self._vvs(buggy)
        return self.__calc_dynamic(b_vari_traces, r_vari_traces)
    
    def __calc_dynamic(self, b_vari_traces:dict, r_vari_traces:dict) -> float:
        similarity = 0
        for no, b_traces in b_vari_traces.items():
            r_traces = r_vari_traces[no]

            var_sim = 0
            for b_var, b_values in b_traces.items():    
                max_var = None
                max_lcs = 0
                for r_var, r_values in r_traces.items():
                    if max_lcs > max(len(b_values), len(r_values)): 
                        continue
                    
                    if b_values == r_values:
                        max_lcs = len(b_values)
                        max_var = r_var
                        break
                    
                    lcs = ETC.calc_lcs(b_values, r_values)
                    if lcs >= max_lcs:
                        max_lcs = lcs
                        max_var = r_var
                
                if max_var is None:
                    continue
                max_values = max(len(b_values), len(r_traces[max_var]))
                var_sim += ETC.divide(max_lcs, max_values)
            similarity += ETC.divide(var_sim, len(b_traces))
        similarity = ETC.divide(similarity, len(b_vari_traces))
        return similarity
    
    
    # [F5] Memory Usage
    def mem_usage(self, buggy:str, references:dict) -> dict:
        results = {}
        b_score = self._memory_usage(buggy)
        for r_id, r_code in tqdm(references.items(), 
                                 total=len(references),
                                 desc="F5", 
                                 leave=False):
            r_score = self._memory_usage(r_code)
            results[r_id] = self.__calc_efficiency(b_score, r_score)
        return results
    
    @cache
    def _memory_usage(self, code:str) -> float:
        results = Tester.trace(code)
        score = 0
        for _, result in results:
            if result.status == Tester.success:
                score += self.__sigmoid(result.mem_usage)
            else:
                score += 1
        return ETC.divide(score, len(results))
    
    def eval_memory(self, buggy:str, reference:str) -> float:
        r_score = self._memory_usage(reference)
        b_score = self._memory_usage(buggy)
        return self.__calc_efficiency(b_score, r_score)
    
    
    # [F6] Execution Time
    def exec_time(self, buggy:str, references:dict) -> dict:
        results = {}
        b_score = self._exec_time(buggy)
        for r_id, r_code in tqdm(references.items(), 
                                 total=len(references),
                                 desc="F6", 
                                 leave=False):
            r_score = self._exec_time(r_code)
            results[r_id] = self.__calc_efficiency(b_score, r_score)
        return results
    
    @cache
    def _exec_time(self, code:str) -> float:
        results = Tester.trace(code)
        score = 0
        for _, result in results:
            if result.status == Tester.success:
                score += self.__sigmoid(result.exec_time)
            else:
                score += 1
        return ETC.divide(score, len(results))
    
    def eval_exec_time(self, buggy:str, reference:str) -> float:
        r_score = self._exec_time(reference)
        b_score = self._exec_time(buggy)
        return self.__calc_efficiency(b_score, r_score)
    
    def __sigmoid(self, x) -> float:
        return 1 / (1 + np.exp(-x))
    
    def __calc_efficiency(self, a_score, b_score) -> float:
        return ETC.divide(1 + (a_score - b_score), 2)
    
    
    def run(self, buggy:str, patch:str) -> dict:
        scores = {}
        for obj in self.objectives:
            scores[obj] = self.OBJ_EVAL_MAP[obj](buggy, patch)
        return scores
    
    def hypervolume(self, scores:dict | list) -> float:
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