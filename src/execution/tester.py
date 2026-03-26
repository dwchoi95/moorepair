import json
import sys
import subprocess
import multiprocessing
import tempfile
import warnings
from decimal import Decimal, InvalidOperation
from functools import cache

from .program import Program
from .results import Result, Results, TestcaseResult
from .testcases import TestCase, TestCases

warnings.filterwarnings("ignore")


class Status:
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"

class TimeLimitExceeded(Exception): pass
class MemoryLimitExceeded(Exception): pass

_SUBPROCESS_RUNNER = r"""
import json
import sys
import time
import tempfile
import tracemalloc
import traceback

class MemoryLimitExceeded(Exception):
    pass


def main():
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:
        sys.stdout.write(json.dumps({
            "status": "error",
            "stdout": "",
            "stderr": f"Runner payload parse error: {exc}",
            "profile": {},
            "memory": 0.0,
        }))
        return

    code = payload.get("code", "")
    input_tc = payload.get("input", "")
    memlimit = float(payload.get("memlimit", 64))
    profiling = bool(payload.get("profiling", False))

    profile = {}
    code_lines = code.splitlines()
    state = {"prev_line": None, "prev_time": None}

    def _finalize(lineno, now, mem):
        prev_time = state["prev_time"]
        if prev_time is None:
            return
        runtime = now - prev_time
        entry = profile.setdefault(str(lineno), {"hits": 0, "runtime": 0.0, "memory": 0.0, "statement": ""})
        entry["hits"] += 1
        entry["runtime"] += runtime
        entry["memory"] = mem
        entry["statement"] = code_lines[lineno - 1] if 0 < lineno <= len(code_lines) else ""

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != "<student_code>":
            return tracer
        memory = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
        now = time.process_time()
        if event == "line":
            if state["prev_line"] is not None:
                _finalize(state["prev_line"], now, memory)
            state["prev_line"] = frame.f_lineno
            state["prev_time"] = now
        elif event == "return":
            if state["prev_line"] is not None:
                _finalize(state["prev_line"], now, memory)
                state["prev_line"] = None
                state["prev_time"] = None
        return tracer

    old_stdin = sys.stdin
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    tmp_stdin = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    tmp_stdout = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    tmp_stderr = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    tmp_stdin.write(input_tc)
    tmp_stdin.seek(0)
    sys.stdin = tmp_stdin
    sys.stdout = tmp_stdout
    sys.stderr = tmp_stderr

    memory = 0.0
    runtime = 0.0
    profile = {}
    tracemalloc.start()
    if profiling:
        sys.settrace(tracer)
    compiled = compile(code, "<student_code>", "exec")
    sandbox_globals = {"__name__": "__main__", "__builtins__": __builtins__}
    start = time.process_time()
    try:
        exec(compiled, sandbox_globals)
    except SystemExit:
        pass
    finally:
        runtime = time.process_time() - start
        if profiling:
            sys.settrace(None)
        memory = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        tmp_stdout.seek(0)
        tmp_stderr.seek(0)
        stdout = tmp_stdout.read().strip()
        stderr = tmp_stderr.read().strip()
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tmp_stdin.close()
        tmp_stdout.close()
        tmp_stderr.close()

    response = {
        "stdout": stdout,
        "stderr": stderr,
        "profile": profile,
        "runtime": runtime,
        "memory": memory,
    }
    sys.stdout.write(json.dumps(response))


if __name__ == "__main__":
    main()
"""

class Tester:
    @classmethod
    def init_globals(
        cls,
        testcases: TestCases,
        timelimit: int = 1,
        memlimit: int = 64,
    ):
        cls.testcases = testcases
        cls.timelimit = timelimit + 0.5
        cls.memlimit = memlimit + 1
        cls._run_cache.cache_clear()

    @classmethod
    def tests_split(cls, results: Results) -> tuple[set[TestCase], set[TestCase]]:
        passed, failed = set(), set()
        for testcase_result in results:
            if testcase_result.result.status == Status.PASSED:
                passed.add(testcase_result.testcase)
            else:
                failed.add(testcase_result.testcase)
        return passed, failed

    @classmethod
    def is_all_pass(cls, results: Results) -> bool:
        return all(testcase_result.result.status == Status.PASSED for testcase_result in results)
    
    @classmethod
    def __is_equal(cls, expect: str, stdout: str) -> bool:
        exp_toks = expect.strip().split()
        out_toks = stdout.strip().split()
        if len(exp_toks) != len(out_toks):
            return False
        return all(cls._token_equal(a, b) for a, b in zip(exp_toks, out_toks))

    @classmethod
    def _token_equal(cls, a: str, b: str) -> bool:
        va = cls._parse_value(a)
        vb = cls._parse_value(b)
        da = cls._to_decimal(va)
        db = cls._to_decimal(vb)
        if da is not None and db is not None:
            return da == db
        return a == b

    @classmethod
    def _parse_value(cls, s: str):
        lowered = s.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            if "." not in s and "e" not in lowered:
                return int(s, 10)
        except ValueError:
            pass
        try:
            return Decimal(s)
        except (InvalidOperation, ValueError):
            return s

    @classmethod
    def _to_decimal(cls, value):
        if isinstance(value, bool):
            return Decimal(1 if value else 0)
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, Decimal):
            return value
        return None
    
    
    @classmethod
    def _profiler(cls, code: str, input_tc: str, profiling: bool) -> tuple:
        payload = json.dumps(
            {
                "code": code,
                "input": input_tc,
                "memlimit": cls.memlimit,
                "profiling": profiling,
            },
            ensure_ascii=False,
        )

        status = None
        stdout = stderr = ""
        runtime = cls.timelimit
        memory = cls.memlimit
        profile = {}
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                completed = subprocess.run(
                    [sys.executable, "-c", _SUBPROCESS_RUNNER],
                    input=payload,
                    text=True,
                    capture_output=True,
                    timeout=cls.timelimit*100 if profiling else cls.timelimit,
                    check=False,
                    cwd=tmpdir,
                )
                raw_response = completed.stdout.strip()
                response = json.loads(raw_response)
                stdout = str(response.get("stdout", ""))
                stderr = str(response.get("stderr", ""))
                runtime = float(response.get("runtime", 0.0))
                memory = float(response.get("memory", 0.0))
                raw_profile = response.get("profile", {})
                profile = {}
                if isinstance(raw_profile, dict):
                    for lineno, value in raw_profile.items():
                        try:
                            profile[int(lineno)] = value
                        except (TypeError, ValueError):
                            continue
        except Exception as exc:
            status = Status.ERROR

        return status, stdout, stderr, profile, runtime, memory

    @classmethod
    def _validation(cls, args:tuple[str, TestCase, bool]) -> TestcaseResult:
        code, tc, profiling = args
        status, stdout, stderr, profile, runtime, memory = \
            cls._profiler(code, tc.input, profiling)
        if status is None:
            if cls.__is_equal(tc.output, stdout):
                status = Status.PASSED
            else:
                status = Status.FAILED
        return TestcaseResult(
            testcase=tc,
            result=Result(
                status=status,
                stdout=stdout,
                stderr=stderr,
                runtime=runtime,
                memory=memory,
                profile=profile
            )
        )
    
    @classmethod
    @cache
    def _run_cache(cls, code: str, profiling: bool = False) -> Results:
        args = [(code, tc, profiling) for tc in cls.testcases]
        processes = min(len(args), multiprocessing.cpu_count())
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(cls._validation, args)
        return Results(results)

    @classmethod
    def run(cls, program: Program, profiling: bool = False) -> Results:
        if program.results is not None:
            return program.results
        program.results = cls._run_cache(program.code, profiling)
        return program.results
