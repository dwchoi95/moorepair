from dataclasses import dataclass, field

from .testcases import TestCase


@dataclass
class Result:
    status:str = field(metadata={"desc":"Status of the execution (e.g., passed, failed, timeout, error)"})
    stdout:str = field(metadata={"desc":"Standard output from the execution"})
    stderr:str = field(metadata={"desc":"Standard error from the execution"})
    returncode:int = field(metadata={"desc":"Return code from the execution"})
    exec_time:float = field(default=0.0, metadata={"desc":"Execution time in seconds"})
    mem_usage:float = field(default=0.0, metadata={"desc":"Memory usage in megabytes"})

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
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Results(self.ts[idx])
        return self.ts[idx]
    
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
    
    def exec_time(self) -> float:
        total_exec_time = 0.0
        for tr in self.ts:
            total_exec_time += tr.result.exec_time
        return total_exec_time
    
    def mem_usage(self) -> float:
        total_mem_usage = 0.0
        for tr in self.ts:
            total_mem_usage += tr.result.mem_usage
        return total_mem_usage
    
    