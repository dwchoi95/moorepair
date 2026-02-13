import re
import statistics
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
    coverage:list[int] = field(default_factory=list, metadata={"desc":"Covered source line numbers for this testcase"})

@dataclass
class TestcaseResult:
    testcase: TestCase
    result: Result = None
        
class Results:
    _INLINE_COMMENT_RE = re.compile(r"//.*$")
    _BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/")
    _TRIVIAL_STMT_RE = re.compile(r"^[{};]+$")

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
        total_exec_time = []
        for tr in self.ts:
            total_exec_time.append(tr.result.exec_time)
        score = sum(total_exec_time)
        return score
    
    def mem_usage(self) -> float:
        total_mem_usage = []
        for tr in self.ts:
            total_mem_usage.append(tr.result.mem_usage)
        score = sum(total_mem_usage)
        return score
    
    def coverage(self) -> set:
        cov = list()
        for tr in self.ts:
            lines = tr.result.coverage if (tr.result and tr.result.coverage) else []
            cov.extend(lines)
        return set(cov)

    def get_coverage_line(self) -> dict:
        tc_coverage = {}
        for tr in self.ts:
            lines = tr.result.coverage if (tr.result and tr.result.coverage) else []
            tc_coverage[tr.testcase.id] = set(lines)
        return tc_coverage

    @classmethod
    def _normalize_stmt(cls, stmt:str) -> str:
        if not stmt:
            return ""
        stmt = cls._BLOCK_COMMENT_RE.sub(" ", stmt)
        stmt = cls._INLINE_COMMENT_RE.sub("", stmt)
        stmt = " ".join(stmt.split())
        if not stmt:
            return ""
        if cls._TRIVIAL_STMT_RE.match(stmt):
            return ""
        return stmt

    def get_coverage_stmt(self, code:str) -> dict:
        code_lines = code.splitlines()
        line_cov = self.get_coverage_line()
        tc_stmts = {}
        for tc_id, line_nums in line_cov.items():
            stmts = set()
            for lineno in line_nums:
                if not isinstance(lineno, int):
                    continue
                if lineno < 1 or lineno > len(code_lines):
                    continue
                stmt = self._normalize_stmt(code_lines[lineno - 1])
                if stmt:
                    stmts.add(stmt)
            tc_stmts[tc_id] = stmts
        return tc_stmts
