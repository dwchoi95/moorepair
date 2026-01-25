import warnings
warnings.filterwarnings("ignore")

import os
import time
import signal
import psutil
import threading
import subprocess
import tempfile
from functools import cache
from multiprocess import Process, Queue

from .program import Program
from .testcases import TestCases
from .results import Result, TestcaseResult, Results

class Tester:
    @classmethod
    def init_globals(cls, testcases:list, timelimit:int=1):
        cls.testcases = TestCases(testcases)
        cls.timelimit = timelimit
    
    @classmethod
    def clear_cache(cls):
        cls.__run.cache_clear()
    
    @classmethod
    def tests_split(cls, program:Program) -> tuple:
        passed, failed = [], []
        for tr in program.results:
            if tr.result.status == "passed":
                passed.append(tr.testcase)
            else:
                failed.append(tr.testcase)
        return passed, failed
    
    @classmethod
    def is_all_pass(cls, program:Program) -> bool:
        return all(tr.result.status == "passed" for tr in program.results)
        
    @classmethod
    def __run_process(cls, q, cmd, stdin, cwd):
        start = time.perf_counter()
        
        p = subprocess.Popen(
            cmd, cwd=cwd, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        proc = psutil.Process(p.pid)
        mem_usage = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                mem_usage += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass

        status, out, err = "", "", ""
        rc = -1
        try:
            out, err = p.communicate(input=stdin, timeout=cls.timelimit)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            p.kill()
            status = "timeout"
            out, err = p.communicate()

        exec_time = time.perf_counter() - start
        q.put((status, rc, out, err, exec_time, mem_usage))

    @classmethod
    def _run(cls, cmd:list[str], *, cwd:str, stdin:str="") -> Result:
        q = Queue()
        proc = Process(
            target=cls.__run_process,
            args=(q, cmd, stdin, cwd),
        )
        proc.start()
    
        try:
            status, rc, out, err, exec_time, mem_usage = q.get()
            proc.join()
        except Exception as e:
            status = "error"
            rc = -1
            out = ""
            err = str(e)
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join()
        
        return Result(
                status=status,
                stdout=out,
                stderr=err,
                exit_code=rc,
                runtime=exec_time,
                memory=mem_usage
            )
    
    
    @classmethod
    def _validation(cls, res:Result, exe:list, td:str) -> Results:
        # Validation
        ts = []
        for tc in cls.testcases:
            # Compilation Error
            if res.exit_code != 0:
                ts.append(TestcaseResult(testcase=tc, result=res))
                continue
            res = cls._run(exe, cwd=td, stdin=tc.input)
            passed = (res.stdout.strip() == tc.output.strip()) and res.exit_code == 0
            res.status = "passed" if passed else "failed"
            ts.append(TestcaseResult(testcase=tc, result=res))
        return Results(ts)

    @classmethod
    def run_cpp(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.cpp")
            exe = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run(["g++", "-std=c++17", "-O2", "-pipe", src, "-o", exe], cwd=td)
            return cls._validation(res, [exe], td)

    @classmethod
    def run_c(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.c")
            exe = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run(["gcc", "-O2", "-pipe", src, "-o", exe], cwd=td)
            return cls._validation(res, [exe], td)
            
    @classmethod
    def run_java(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, f"{program.mn}.java")
            exe = ["java", "-cp", td, program.mn]
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run(["javac", f"{program.mn}.java"], cwd=td)
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
                    exit_code=-1,
                )
            exe = ["python3", "main.py"]
            return cls._validation(res, [exe], td)
    
    @classmethod
    def run_javascript(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.js")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = Result(status="", stdout="", stderr="", exit_code=0, timed_out=False)
            exe = ["node", "main.js"]
            return cls._validation(res, [exe], td)

    @classmethod
    def run_r(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.R")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = Result(status="", stdout="", stderr="", exit_code=0, timed_out=False)
            exe = ["Rscript", "main.R"]
            return cls._validation(res, [exe], td)

    @classmethod
    def run_csharp(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, f"{program.mn}.cs")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            exe_path = os.path.join(td, "Main.exe")
            res = cls._run(["mcs", "-optimize+", "-out:Main.exe", f"{program.mn}.cs"], cwd=td)
            exe = ["mono", exe_path]
            return cls._validation(res, [exe], td)
        
    @classmethod
    def run_go(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.go")
            exe_path = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run(["go", "build", "-o", exe_path, "main.go"], cwd=td)
            exe = [exe_path]
            return cls._validation(res, [exe], td)
        
    @classmethod
    def run_rust(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.rs")
            exe_path = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run(["rustc", "-O", "main.rs", "-o", exe_path], cwd=td)
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
        program.results = cls.__run(program)
        return program.results
