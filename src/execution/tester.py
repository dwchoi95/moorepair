import warnings
warnings.filterwarnings("ignore")

import re
import os
import glob
import shutil
import time
import psutil
import threading
import subprocess
import tempfile
from functools import cache
from decimal import Decimal, InvalidOperation
from multiprocessing import Process, Queue

from .program import Program
from .testcases import TestCases, TestCase
from .results import Result, TestcaseResult, Results

class Tester:
    _INT_RE = re.compile(r"^[+-]?\d+$")
    _FLOAT_RE = re.compile(
        r"^[+-]?("
        r"(\d+\.\d*)|(\d*\.\d+)|(\d+)"
        r")([eE][+-]?\d+)?$"
    )
    
    @classmethod
    def init_globals(cls, testcases:TestCases, timelimit:int=1, collect_coverage:bool=True):
        cls.testcases = testcases
        cls.timelimit = timelimit
        cls.collect_coverage = collect_coverage
    
    @classmethod
    def clear_cache(cls):
        cls.__run.cache_clear()
    
    @classmethod
    def tests_split(cls, results:Results) -> tuple[set[TestCase], set[TestCase]]:
        passed, failed = set(), set()
        for tr in results:
            if tr.result.status == "passed":
                passed.add(tr.testcase)
            else:
                failed.add(tr.testcase)
        return passed, failed
    
    @classmethod
    def is_all_pass(cls, results:Results) -> bool:
        return all(tr.result.status == "passed" for tr in results)

    @classmethod
    def _run_compile(cls, cmd:list[str], *, cwd:str) -> Result:
        """Run compilation command and return the result."""
        status = stdout = stderr = ""
        returncode = -1
        exec_time = 0
        mem_usage = 0
        
        try:
            p = subprocess.Popen(
                cmd,
                cwd=cwd,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            start = time.perf_counter()
            
            try:
                stdout, stderr = p.communicate(timeout=cls.timelimit)
                exec_time = time.perf_counter() - start
                returncode = p.returncode
                status = "ok"
            
            except subprocess.TimeoutExpired:
                p.kill()
                stdout, stderr = p.communicate()
                exec_time = cls.timelimit
                status = "timeout"
            
        except Exception as e:
            status = "error"
            stderr = str(e)
        
        return Result(
            status=status,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            exec_time=exec_time,
            mem_usage=mem_usage
        )

    @classmethod
    def _run_test(
        cls,
        cmd:list[str],
        *,
        cwd:str,
        tc:TestCase,
        compile_result:Result=None,
        collect_coverage:bool=False,
        coverage_source:str|None=None,
        coverage_cwd:str|None=None,
        env:dict|None=None
    ) -> TestcaseResult:
        """Execute a single test case."""
        # If compilation failed, reuse the compilation error for all test cases
        if compile_result is not None and compile_result.returncode != 0:
            return TestcaseResult(testcase=tc, result=compile_result)
        
        status = stdout = stderr = ""
        returncode = -1
        exec_time = cls.timelimit
        mem_usage = 0 # in bytes
        
        def monitor(proc:psutil.Process):
            nonlocal mem_usage
            while True:
                try:
                    mem = proc.memory_info().rss
                    try:
                        for child in proc.children(recursive=True):
                            mem += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    mem_usage = max(mem_usage, mem)
                except Exception:
                    break
                time.sleep(0.001)
    
        try:
            p = subprocess.Popen(
                cmd,
                cwd=cwd,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            start = time.perf_counter()
            proc = psutil.Process(p.pid)
            
            t = threading.Thread(target=monitor, args=(proc,), daemon=True)
            t.start()
            
            try:
                stdout, stderr = p.communicate(input=tc.input, timeout=cls.timelimit)
                exec_time = time.perf_counter() - start
                returncode = p.returncode
                status = "ok"
            
            except subprocess.TimeoutExpired:
                p.kill()
                stdout, stderr = p.communicate()
                exec_time = cls.timelimit
                status = "timeout"
            
            finally:
                t.join(timeout=1)
            
        except Exception as e:
            status = "error"
            stderr = str(e)
        
        result = Result(
            status=status,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            exec_time=exec_time,
            mem_usage=mem_usage
        )

        if collect_coverage and coverage_source:
            cov_cwd = coverage_cwd if coverage_cwd else cwd
            result.coverage = cls._collect_testcase_coverage(cov_cwd, coverage_source)
        
        # Check if test passed
        passed = cls.__is_equal(tc.output, result.stdout) and result.returncode == 0
        result.status = "passed" if passed else "failed"
        
        return TestcaseResult(testcase=tc, result=result)

    @classmethod
    def __is_equal(cls, expect:str, stdout:str) -> bool:
        exp_toks = expect.split()
        out_toks = stdout.split()
        if len(exp_toks) != len(out_toks):
            return False

        for a, b in zip(exp_toks, out_toks):
            if not cls._token_equal(a, b):
                return False
        return True
    
    @classmethod
    def _token_equal(cls, a:str, b:str) -> bool:
        va = cls._parse_value(a)
        vb = cls._parse_value(b)
        da = cls._to_decimal(va)
        db = cls._to_decimal(vb)
        if da is not None and db is not None:
            return da == db
        return a == b

    @classmethod
    def _parse_value(cls, s:str):
        sl = s.strip().lower()
        if sl == "true":
            return True
        if sl == "false":
            return False
        if cls._INT_RE.match(s):
            try:
                return int(s, 10)
            except ValueError:
                return s
        if cls._FLOAT_RE.match(s):
            try:
                return Decimal(s)
            except (InvalidOperation, ValueError):
                return s
        return s

    @classmethod
    def _to_decimal(cls, v):
        if isinstance(v, bool):
            return Decimal(1 if v else 0)
        if isinstance(v, int):
            return Decimal(v)
        if isinstance(v, Decimal):
            return v
        return None
        
    
    @classmethod
    def _clear_coverage_artifacts(cls, cwd:str) -> None:
        for pattern in ("*.gcda", "*.gcov"):
            for path in glob.glob(os.path.join(cwd, pattern)):
                try:
                    os.remove(path)
                except OSError:
                    pass

    @classmethod
    def _collect_testcase_coverage(cls, cwd:str, coverage_source:str) -> list[int]:
        cls._run_compile(["gcov", coverage_source], cwd=cwd)
        gcov_path = os.path.join(cwd, f"{coverage_source}.gcov")
        covered = cls._parse_gcov_covered_lines(gcov_path)
        return sorted(covered)

    @classmethod
    def _run_test_worker(
        cls,
        cmd:list[str],
        *,
        cwd:str,
        tc:TestCase,
        queue:Queue,
        compile_result:Result=None,
        collect_coverage:bool=False,
        coverage_source:str|None=None,
        coverage_cwd:str|None=None,
        env:dict|None=None
    ) -> None:
        tr = cls._run_test(
            cmd,
            cwd=cwd,
            tc=tc,
            compile_result=compile_result,
            collect_coverage=collect_coverage,
            coverage_source=coverage_source,
            coverage_cwd=coverage_cwd,
            env=env
        )
        queue.put(tr)

    @classmethod
    def _run_test_isolated_coverage_worker(
        cls,
        cmd:list[str],
        *,
        base_cwd:str,
        tc:TestCase,
        queue:Queue,
        compile_result:Result,
        coverage_source:str
    ) -> None:
        # If compilation failed, reuse the compilation error for all test cases
        if compile_result is not None and compile_result.returncode != 0:
            queue.put(TestcaseResult(testcase=tc, result=compile_result))
            return

        case_dir = tempfile.mkdtemp(prefix=f"tc_{tc.id}_", dir=base_cwd)
        try:
            # Prepare gcov inputs in testcase-local directory.
            for name in os.listdir(base_cwd):
                src = os.path.join(base_cwd, name)
                if not os.path.isfile(src):
                    continue
                if name == coverage_source or name.endswith(".gcno"):
                    shutil.copy2(src, os.path.join(case_dir, name))

            # Redirect gcda emission to testcase-local directory.
            strip = len([p for p in os.path.normpath(base_cwd).split(os.sep) if p])
            run_env = os.environ.copy()
            run_env["GCOV_PREFIX"] = case_dir
            run_env["GCOV_PREFIX_STRIP"] = str(strip)

            tr = cls._run_test(
                cmd,
                cwd=base_cwd,
                tc=tc,
                compile_result=compile_result,
                collect_coverage=True,
                coverage_source=coverage_source,
                coverage_cwd=case_dir,
                env=run_env
            )
            queue.put(tr)
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    @classmethod
    def _validation(
        cls,
        compile_result:Result,
        exe:list,
        td:str,
        collect_coverage:bool=False,
        coverage_source:str|None=None
    ) -> Results:
        """Validate program by running all test cases with multiprocessing."""
        if compile_result is not None and compile_result.returncode != 0:
            results = [TestcaseResult(testcase=tc, result=compile_result) for tc in cls.testcases]
            results.sort(key=lambda tr: tr.testcase.id)
            return Results(results)

        results = []
        should_collect_coverage = bool(collect_coverage and coverage_source)
        queue = Queue()
        processes = []

        for tc in cls.testcases:
            if should_collect_coverage:
                p = Process(
                    target=cls._run_test_isolated_coverage_worker,
                    args=(exe,),
                    kwargs={
                        "base_cwd": td,
                        "tc": tc,
                        "queue": queue,
                        "compile_result": compile_result,
                        "coverage_source": coverage_source
                    }
                )
            else:
                p = Process(
                    target=cls._run_test_worker,
                    args=(exe,),
                    kwargs={
                        "cwd": td,
                        "tc": tc,
                        "queue": queue,
                        "compile_result": compile_result,
                        "collect_coverage": False,
                        "coverage_source": None
                    }
                )
            p.start()
            processes.append(p)

        for _ in range(len(cls.testcases)):
            results.append(queue.get())

        for p in processes:
            p.join()
        
        # Sort results by test case id to maintain order
        results.sort(key=lambda tr: tr.testcase.id)
        
        return Results(results)

    @classmethod
    def _parse_gcov_covered_lines(cls, gcov_path:str) -> set[int]:
        covered = set()
        if not os.path.isfile(gcov_path):
            return covered
        with open(gcov_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue
                count = parts[0].strip()
                lineno = parts[1].strip()
                if not lineno.isdigit():
                    continue
                if count in {"-", "#####", "====="}:
                    continue
                count = count.replace("*", "")
                if count.isdigit() and int(count) > 0:
                    covered.add(int(lineno))
        return covered

    @classmethod
    def run_cpp(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.cpp")
            exe = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run_compile(["g++", "-std=c++17", "-O2", "-pipe", src, "-o", exe], cwd=td)
            return cls._validation(res, [exe], td)

    @classmethod
    def run_c(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.c")
            exe = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            collect_coverage = getattr(cls, "collect_coverage", False)
            if collect_coverage:
                compile_cmd = ["gcc", "--coverage", "-O0", "-pipe", "main.c", "-o", "main"]
            else:
                compile_cmd = ["gcc", "-O2", "-pipe", src, "-o", exe]
            res = cls._run_compile(compile_cmd, cwd=td)
            return cls._validation(
                res, [exe], td,
                collect_coverage=collect_coverage,
                coverage_source="main.c" if collect_coverage else None
            )
            
    @classmethod
    def run_java(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, f"{program.mn}.java")
            exe = ["java", "-cp", td, program.mn]
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run_compile(["javac", f"{program.mn}.java"], cwd=td)
            return cls._validation(res, [exe], td)
    
    @classmethod
    def run_python(cls, program:Program):
        import ast
        
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.py")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            try: ast.parse(program.code)
            except Exception as e:
                res = Result(
                    status="error",
                    stdout="",
                    stderr=str(e),
                    returncode=-1,
                )
            exe = ["python3", "main.py"]
            return cls._validation(res, [exe], td)
    
    @classmethod
    def run_javascript(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.js")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = Result(status="", stdout="", stderr="", returncode=0, timed_out=False)
            exe = ["node", "main.js"]
            return cls._validation(res, [exe], td)

    @classmethod
    def run_r(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.R")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = Result(status="", stdout="", stderr="", returncode=0, timed_out=False)
            exe = ["Rscript", "main.R"]
            return cls._validation(res, [exe], td)

    @classmethod
    def run_csharp(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, f"{program.mn}.cs")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            exe_path = os.path.join(td, "Main.exe")
            res = cls._run_compile(["mcs", "-optimize+", "-out:Main.exe", f"{program.mn}.cs"], cwd=td)
            exe = ["mono", exe_path]
            return cls._validation(res, [exe], td)
        
    @classmethod
    def run_go(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.go")
            exe_path = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run_compile(["go", "build", "-o", exe_path, "main.go"], cwd=td)
            exe = [exe_path]
            return cls._validation(res, [exe], td)
        
    @classmethod
    def run_rust(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.rs")
            exe_path = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run_compile(["rustc", "-O", "main.rs", "-o", exe_path], cwd=td)
            exe = [exe_path]
            return cls._validation(res, [exe], td)
    
    @classmethod
    @cache
    def __run(cls, program:Program) -> Results:
        ext = program.ext.lower()
        if ext == "c":
            results = cls.run_c(program)
        elif ext == "cpp":
            results = cls.run_cpp(program)
        elif ext == "csharp":
            results = cls.run_csharp(program)
        elif ext == "java":
            results = cls.run_java(program)
        elif ext == "py":
            results = cls.run_python(program)
        elif ext == "js":
            results = cls.run_javascript(program)
        elif ext == "r":
            results = cls.run_r(program)
        elif ext == "go":
            results = cls.run_go(program)
        elif ext == "rs":
            results = cls.run_rust(program)
        else:
            raise ValueError(f"Unsupported language: {ext}")
        return results
    
    @classmethod
    def run(cls, program:Program) -> Results:
        if program.results is None:
            program.results = cls.__run(program)
        return program.results
