from dataclasses import dataclass, field

from .testcases import TestCase


@dataclass
class Result:
    status:str = field(metadata={"desc":"Status of the execution (e.g., passed, failed, timeout, error)"})
    stdout:str = field(metadata={"desc":"Standard output from the execution"})
    stderr:str = field(metadata={"desc":"Standard error from the execution"})
    exit_code:int = field(metadata={"desc":"Exit code of the execution"})
    runtime:float = field(default=0.0, metadata={"desc":"Execution time in seconds"})
    memory:float = field(default=0.0, metadata={"desc":"Memory usage in megabytes"})

@dataclass
class TestcaseResult:
    testcase: TestCase
    result: Result = None
        
class Results:
    def __init__(self, ts:list[TestcaseResult]=[]):
        self.ts = ts
        self.current_index = 0
        
    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index < len(self.ts):
            tr = self.ts[self.current_index]
            self.current_index += 1
            return tr
        raise StopIteration
    
    def __len__(self):
        return len(self.ts)
    
    def __str__(self):
        prints = "|  #  | Input | Expected Output | Actual Output | Status |\n| :-: | :---: | :------: | :----: | :----: |\n"
        for tr in self.ts:
            prints += self.__print(tr) + '\n\n'
        return prints.strip()
    
    def __print(self, tr:TestcaseResult) -> str:
        br = '<br />'
        tc_input = tr.testcase.input.replace('\n', br)
        tc_output = tr.testcase.output.replace('\n', br)
        result_stdout = str(tr.result.stdout).replace('\n', br)
        return f"| {tr.testcase.id} | {tc_input} | {tc_output} | {result_stdout} | {tr.result.status} |"
    
    def update(self, testcase:TestCase, result:Result):
        for tr in self.ts:
            if tr.testcase.id == testcase.id:
                tr.result = result
                return
        self.ts.append(TestcaseResult(testcase=testcase, result=result))
    
    def delete(self, testcase:TestCase):
        self.ts = [tr for tr in self.ts if tr.testcase.id != testcase.id]
    
    def runtime(self) -> float:
        total_runtime = 0.0
        for tr in self.ts:
            total_runtime += tr.result.runtime
        return total_runtime
    
    def memory(self) -> float:
        total_memory = 0.0
        for tr in self.ts:
            total_memory += tr.result.memory
        return total_memory
    
    