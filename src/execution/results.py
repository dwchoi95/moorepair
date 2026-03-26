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
    
    def time_report(self) -> str:
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

        total_runtime = sum(e["runtime"] for e in self.profile.values())
        sorted_lines = sorted(self.profile.keys(), key=lambda k: int(k))

        header = (
            f"Total runtime: {fmt_time(self.runtime)}\n\n"
            f"{'Line #':>8}  {'Hits':>6}  {'Time':>12}  {'Per Hit':>12}  "
            f"{'% Time':>8}  Line Contents\n"
            f"{'=' * 80}"
        )
        lines = [header]

        for lineno in sorted_lines:
            entry = self.profile[lineno]
            hits = entry["hits"]
            runtime = entry["runtime"]
            stmt = entry.get("statement", "")
            per_hit = runtime / hits if hits else 0.0
            pct_time = (runtime / total_runtime * 100) if total_runtime else 0.0
            lines.append(
                f"{lineno:>8}  {hits:>6d}  {fmt_time(runtime):>12}  {fmt_time(per_hit):>12}  "
                f"{pct_time:>7.1f}%  {stmt}"
            )

        return "\n".join(lines)

    def mem_report(self) -> str:
        if not self.profile:
            return ""

        def fmt_mem(m):
            if m < 0.001:
                return f"{m * 1024 * 1024:.1f} B"
            if m < 1.0:
                return f"{m * 1024:.1f} KiB"
            return f"{m:.2f} MiB"

        sorted_lines = sorted(self.profile.keys(), key=lambda k: int(k))

        header = (
            f"Total memory: {fmt_mem(self.memory)}\n\n"
            f"{'Line #':>8}  {'Mem usage':>12}  {'Increment':>12}  {'Occurrences':>6}  Line Contents\n"
            f"{'=' * 75}"
        )
        lines = [header]
        prev_mem = 0.0

        for lineno in sorted_lines:
            entry = self.profile[lineno]
            hits = entry["hits"]
            memory = entry["memory"]
            stmt = entry.get("statement", "")
            increment = memory - prev_mem
            lines.append(
                f"{lineno:>8}  {fmt_mem(memory):>12}  {fmt_mem(increment):>12}  {hits:>6d}  {stmt}"
            )
            prev_mem = memory

        return "\n".join(lines)
        
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
        prints = ""
        for tr in self.ts:
            prints += self.__print(tr) + '\n\n'
        return prints.strip()
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Results(self.ts[idx])
        return self.ts[idx]
    
    def __print(self, tr:TestcaseResult) -> str:
        from .tester import Status
        prints = f"[Input]\n{tr.testcase.input}\n[/Input]\n\n"
        prints += f"[Expected]\n{tr.testcase.output}\n[/Expected]\n\n"
        prints += f"[Actual]\n{tr.result.stdout if tr.result != Status.ERROR else tr.result.stderr}\n[/Actual]"
        return prints
    
    def print_tc_result(self, tc:TestCase) -> str:
        for tr in self.ts:
            if tr.testcase.id == tc.id:
                return self.__print(tr)
        return f"No result found for TestCase ID: {tc.id}"

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

    # EffiLearner metrics (ET, MU, TMU)
    def ET(self) -> float:
        """Execution Time: total execution time across all test cases (s)."""
        return self.exec_time()

    def MU(self) -> float:
        """Max Memory Usage: peak memory across all test cases (MB)."""
        return self.mem_usage_max()

    def TMU(self) -> float:
        """Total Memory Usage: sum of memory * runtime across all test cases (MB*s)."""
        return sum(tr.result.memory * tr.result.runtime for tr in self.ts if tr.result)
    
    def report_time(self, tc:TestCase|None=None) -> str:
        max_runtime = 0.0
        max_runtime_tr = None
        for tr in self.ts:
            if tr.testcase.id == tc.id:
                return tr.result.time_report()
            if tr.result and tr.result.runtime > max_runtime:
                max_runtime = tr.result.runtime
                max_runtime_tr = tr
        if max_runtime_tr:
            return max_runtime_tr.result.time_report()
        return f"No time report found for TestCase ID: {tc.id}"

    def report_mem(self, tc:TestCase|None=None) -> str:
        max_memory = 0.0
        max_memory_tr = None
        for tr in self.ts:
            if tr.testcase.id == tc.id:
                return tr.result.mem_report()
            if tr.result and tr.result.memory > max_memory:
                max_memory = tr.result.memory
                max_memory_tr = tr
        if max_memory_tr:
            return max_memory_tr.result.mem_report()
        return f"No memory report found for TestCase ID: {tc.id}"
    