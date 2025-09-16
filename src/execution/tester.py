import warnings
warnings.filterwarnings("ignore")
from functools import cache
from multiprocessing import Process, Queue

from .results import Result, Results
from .testsuite import TestSuite, Testcase
from .unittests import RunUnitTest, UnitTestStatus, Validating, Tracing
from ..utils import NodeParser, Regularize

    
class Tester:
    success = UnitTestStatus.success
    error = UnitTestStatus.error
    failure = UnitTestStatus.failure
    
    @classmethod
    def init_globals(cls, 
                 testcases:list,
                 timeLimit:int=1,
                 memLimit:int=256,
                 title:str="",
                 description:str=""):
        # arguments
        cls.testsuite = TestSuite(testcases)
        cls.timeLimit = timeLimit
        cls.memLimit = memLimit
        cls.title = title
        cls.description = description
    
    @classmethod
    def clear_cache(cls):
        cls.test.cache_clear()
        cls.trace.cache_clear()
        
    @classmethod
    def split_test_hist(cls, test_hist:dict) -> tuple[list, list]:
        passed_tc_list = []
        failed_tc_list = []
        for tc_id, result in test_hist.items():
            if result == cls.success:
                passed_tc_list.append(tc_id)
            else:
                failed_tc_list.append(tc_id)
        return passed_tc_list, failed_tc_list
    
    @classmethod
    def is_all_fail(cls, test_hist:dict) -> bool:
        return all(result != cls.success 
                   for result in test_hist.values())
    
    @classmethod
    def is_all_pass(cls, test_hist:dict) -> bool:
        return all(result == cls.success 
                   for result in test_hist.values())
    
    @classmethod
    def is_pass(cls, status:str) -> bool:
        return status == cls.success
    
    @classmethod
    def seperate_results(cls, results:Results) -> tuple[dict, dict, dict]:
        test_hist = {}
        vari_hist = {}
        exec_hist = {}
        for no, result in results:
            test_hist[no] = result.status
            vari_hist[no] = result.vari_traces
            exec_hist[no] = result.exec_traces
        return test_hist, vari_hist, exec_hist
    
    @classmethod
    def print_test_results(cls, results:Results) -> str:
        test_results = "|  #  | Input | Expected | Actual | Status |\n| :-: | :---: | :------: | :----: | :----: |\n"
        for no, result in results:
            test_input = result.input.replace("\n", "<br />")
            test_expect = result.expect.replace("\n", "<br />")
            test_actual = str(result.stdout).replace("\n", "<br />")
            test_results += f"|  {no}  | {test_input} | {test_expect} | {test_actual} | {result.status} |\n"
        return test_results.strip()
    
    @classmethod
    def get_test_hist(cls, results:Results) -> dict:
        test_hist = {}
        for no, result in results:
            test_hist[no] = result.status
        return test_hist
    
    @classmethod
    def get_vari_hist(cls, results:Results) -> dict:
        vari_hist = {}
        for no, result in results:
            vari_hist[no] = result.vari_traces
        return vari_hist
    
    @classmethod
    def get_exec_hist(cls, results:Results) -> dict:
        exec_hist = {}
        for no, result in results:
            exec_hist[no] = result.exec_traces
        return exec_hist
    
    @classmethod
    def gen_test_code(cls, code:str, input:str, 
                      hasKeyboardInput:bool=False) -> str:
        test_code = code.strip()
        if not hasKeyboardInput:
            if 'print(' not in input:
                input = 'print(' + input + ')'
            test_code = code + '\n\n' + input
        return test_code
   
    @classmethod     
    def __eval_vari_values(cls):
        for var, vvs_dict in Result.vari_traces.items():
            new_vvs = {}
            for lineno, value in vvs_dict.items():
                try: value = eval(value)
                except: pass
                new_vvs[lineno] = value
            Result.vari_traces[var] = new_vvs
    
    @classmethod
    def run_core(cls, code:str, testcase:Testcase, UnitTest:object=Validating, q:Queue=None) -> Result:
        # Preprocess
        code = code.replace('sys.stdin.buffer', 'sys.stdin')
        if UnitTest == Tracing:
            code = Regularize.run(code)
            
        np = NodeParser()
        np.run(code)
        Result.init_globals()
        Result.line_vari_map = np.line_vari_map
        Result.timeLimit = cls.timeLimit
        Result.input = testcase.input
        Result.expect = testcase.expect
        Result.test_code = cls.gen_test_code(code, testcase.input, testcase.hasStdIn)
        Result.end_line = len(code.splitlines())
        
        # Debug
        # print(Result.test_code)
        
        # Unittest
        RunUnitTest.run(UnitTest)
        if UnitTest == Tracing:
            cls.__eval_vari_values()
        
        result = Result()
        # print(result.stdout)
        if q is not None:
            q.put((testcase.no, result))
        return result


    @classmethod
    def _run(cls, code:str, UnitTest:object=Validating) -> Results:
        results = Results(code)
        
        # for testcase in cls.testsuite:
        #     results[testcase.no] = \
        #         cls.run_core(code, testcase, UnitTest)
        
        q = Queue()
        procs = [Process(target=cls.run_core,
                        args=(code, testcase, UnitTest, q))
                for testcase in cls.testsuite]
        for proc in procs:
            proc.start()
        for proc in procs:
            no, result = q.get()
            results[no] = result
        for proc in procs:
            proc.join()
          
        return results
    
    @classmethod
    @cache
    def test(cls, code:str) -> Results:
        return cls._run(code, Validating)
    
    @classmethod
    @cache
    def trace(cls, code:str) -> Results:
        return cls._run(code, Tracing)