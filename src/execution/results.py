import re
import statistics
from dataclasses import dataclass, field

from .testcases import TestCase


@dataclass
class Result:
    status:str = field(metadata={"desc":"Status of the execution (e.g., passed, failed, timeout, error)"})
    stdout:str = field(metadata={"desc":"Standard output from the execution"})
    stderr:str = field(metadata={"desc":"Standard error from the execution"})
    runtime:float = field(default=0.0, metadata={"desc":"Execution time in seconds"})
    memory:float = field(default=0.0, metadata={"desc":"Memory usage in megabytes"})
    profile:dict = field(default_factory=dict, metadata={"desc":"Line-level profile: {lineno: {hits, runtime, memory, statement}}"})
    
    def profile_report(self) -> str:
        if not self.profile:
            return ""

        def fmt_time(t):
            if t < 1e-6:
                return f"{t * 1e9:.1f} ns"
            if t < 1e-3:
                return f"{t * 1e6:.1f} µs"
            if t < 1.0:
                return f"{t * 1e3:.1f} ms"
            return f"{t:.4f} s"

        def fmt_mem(m):
            if m < 0.001:
                return f"{m * 1024 * 1024:.1f} B"
            if m < 1.0:
                return f"{m * 1024:.1f} KiB"
            return f"{m:.2f} MiB"

        total_runtime = sum(e["runtime"] for e in self.profile.values())
        sorted_lines = sorted(self.profile.keys(), key=lambda k: int(k))

        header = (
            f"Total runtime: {fmt_time(self.runtime)}    "
            f"Total memory: {fmt_mem(self.memory)}\n\n"
            f"{'Line #':>8}  {'Hits':>6}  {'Time':>12}  {'Per Hit':>12}  "
            f"{'% Time':>8}  {'Mem usage':>12}  {'Increment':>12}  Line Contents\n"
            f"{'=' * 110}"
        )

        lines = [header]
        prev_mem = 0.0
        for lineno in sorted_lines:
            entry = self.profile[lineno]
            hits = entry["hits"]
            runtime = entry["runtime"]
            memory = entry["memory"]
            stmt = entry.get("statement", "")

            per_hit = runtime / hits if hits else 0.0
            pct_time = (runtime / total_runtime * 100) if total_runtime else 0.0
            increment = memory - prev_mem

            lines.append(
                f"{lineno:>8}  {hits:>6d}  {fmt_time(runtime):>12}  {fmt_time(per_hit):>12}  "
                f"{pct_time:>7.1f}%  {fmt_mem(memory):>12}  {fmt_mem(increment):>12}  {stmt}"
            )
            prev_mem = memory

        return "\n".join(lines)

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
            total_exec_time.append(tr.result.runtime)
        score = sum(total_exec_time)
        return score

    def exec_time_max(self) -> float:
        times = [tr.result.runtime for tr in self.ts if tr.result]
        return max(times) if times else 0.0

    def mem_usage(self) -> float:
        total_mem_usage = []
        for tr in self.ts:
            total_mem_usage.append(tr.result.memory)
        score = sum(total_mem_usage)
        return score

    def mem_usage_max(self) -> float:
        mems = [tr.result.memory for tr in self.ts if tr.result]
        return max(mems) if mems else 0.0
    
    def report(self) -> str:
        valid = [tr for tr in self.ts if tr.result and tr.result.profile]
        if not valid:
            return {}

        runtimes = [tr.result.runtime for tr in valid]
        memories = [tr.result.memory for tr in valid]

        rt_min, rt_max = min(runtimes), max(runtimes)
        mem_min, mem_max = min(memories), max(memories)

        rt_range = rt_max - rt_min
        mem_range = mem_max - mem_min

        best_tr = None
        best_score = -1.0

        for tr in valid:
            norm_rt = (tr.result.runtime - rt_min) / rt_range if rt_range else 0.0
            norm_mem = (tr.result.memory - mem_min) / mem_range if mem_range else 0.0
            score = norm_rt + norm_mem
            if score > best_score:
                best_score = score
                best_tr = tr

        return best_tr.result.profile_report()