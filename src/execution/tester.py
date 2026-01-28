import warnings
warnings.filterwarnings("ignore")

import re
import os
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
    def init_globals(cls, testcases:TestCases, timelimit:int=1):
        cls.testcases = testcases
        cls.timelimit = timelimit
    
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
    def _run_test(cls, cmd:list[str], *, cwd:str, tc:TestCase, queue:Queue, compile_result:Result=None) -> None:
        """Execute a test case and put the result in the queue."""
        # If compilation failed, reuse the compilation error for all test cases
        if compile_result is not None and compile_result.returncode != 0:
            queue.put(TestcaseResult(testcase=tc, result=compile_result))
            return
        
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
                stderr=subprocess.PIPE
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
        
        # Check if test passed
        passed = cls.__is_equal(tc.output, result.stdout) and result.returncode == 0
        result.status = "passed" if passed else "failed"
        
        # Put TestcaseResult in queue
        queue.put(TestcaseResult(testcase=tc, result=result))

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
    def _validation(cls, compile_result:Result, exe:list, td:str) -> Results:
        """Validate program by running all test cases in parallel."""
        queue = Queue()
        processes = []
        
        # Start a process for each test case
        for tc in cls.testcases:
            p = Process(target=cls._run_test, args=(exe,), 
                       kwargs={'cwd': td, 'tc': tc, 'queue': queue, 
                              'compile_result': compile_result})
            p.start()
            processes.append(p)
        
        # Collect results from queue
        results = []
        for _ in range(len(cls.testcases)):
            results.append(queue.get())
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Sort results by test case id to maintain order
        results.sort(key=lambda tr: tr.testcase.id)
        
        return Results(results)

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

            res = cls._run_compile(["gcc", "-O2", "-pipe", src, "-o", exe], cwd=td)
            return cls._validation(res, [exe], td)
            
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
