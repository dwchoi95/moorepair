import warnings
warnings.filterwarnings("ignore")

import os
import time
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
        cls.run.cache_clear()
    
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
        ps_p = psutil.Process(p.pid)
        peak_rss = 0

        stop = threading.Event()

        def monitor():
            nonlocal peak_rss
            while not stop.is_set():
                if p.poll() is not None:
                    break
                try:
                    rss = ps_p.memory_info().rss
                    # 자식 프로세스까지 합치고 싶으면 아래 2줄 포함
                    for c in ps_p.children(recursive=True):
                        rss += c.memory_info().rss
                    if rss > peak_rss:
                        peak_rss = rss
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.005)  # 5ms 폴링(너무 촘촘하면 오버헤드)

        t = threading.Thread(target=monitor, daemon=True)
        t.start()

        status = ""
        try:
            out, err = p.communicate(input=stdin, timeout=cls.timelimit)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
            p.kill()
            out, err = p.communicate()
            rc = -1
        finally:
            stop.set()
            t.join(timeout=0.1)

        elapsed = time.perf_counter() - start

        # 메모리 단위: bytes
        peak_rss = peak_rss if peak_rss else 0

        q.put((status, rc, out, err, elapsed, peak_rss))

    @classmethod
    def _run(cls, cmd:list[str], *, cwd:str, stdin:str="") -> Result:
        q = Queue()
        proc = Process(
            target=cls.__run_process,
            args=(q, cmd, stdin, cwd),
        )
        proc.start()
    
        try:
            status, rc, out, err, runtime, memory = q.get(timeout=cls.timelimit+0.5)
            proc.join(timeout=0.1)
                
            return Result(
                status=status,
                stdout=out,
                stderr=err,
                exit_code=rc,
                runtime=runtime,
                memory=memory,
            )
        except Exception as e:
            return Result(
                status="error",
                stdout="",
                stderr=str(e),
                exit_code=-1,
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
            program.results = cls._validation(res, [exe], td)

    @classmethod
    def run_c(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.c")
            exe = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run(["gcc", "-O2", "-pipe", src, "-o", exe], cwd=td)
            program.results = cls._validation(res, [exe], td)
            
    @classmethod
    def run_java(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, f"{program.mn}.java")
            exe = ["java", "-cp", td, program.mn]
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run(["javac", f"{program.mn}.java"], cwd=td)
            program.results = cls._validation(res, [exe], td)
    
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
                    timed_out=False,
                )
            exe = ["python3", "main.py"]
            program.results = cls._validation(res, [exe], td)
    
    @classmethod
    def run_javascript(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.js")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = Result(status="", stdout="", stderr="", exit_code=0, timed_out=False)
            exe = ["node", "main.js"]
            program.results = cls._validation(res, [exe], td)

    @classmethod
    def run_r(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.R")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = Result(status="", stdout="", stderr="", exit_code=0, timed_out=False)
            exe = ["Rscript", "main.R"]
            program.results = cls._validation(res, [exe], td)

    @classmethod
    def run_csharp(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, f"{program.mn}.cs")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            exe_path = os.path.join(td, "Main.exe")
            res = cls._run(["mcs", "-optimize+", "-out:Main.exe", f"{program.mn}.cs"], cwd=td)
            exe = ["mono", exe_path]
            program.results = cls._validation(res, [exe], td)
        
    @classmethod
    def run_go(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.go")
            exe_path = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run(["go", "build", "-o", exe_path, "main.go"], cwd=td)
            exe = [exe_path]
            program.results = cls._validation(res, [exe], td)
        
    @classmethod
    def run_rust(cls, program:Program):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "main.rs")
            exe_path = os.path.join(td, "main")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program.code)

            res = cls._run(["rustc", "-O", "main.rs", "-o", exe_path], cwd=td)
            exe = [exe_path]
            program.results = cls._validation(res, [exe], td)
    
    @classmethod
    @cache
    def run(cls, program:Program):
        ext = program.ext.lower()
        if ext == "c":
            cls.run_c(program)
        elif ext == "cpp":
            cls.run_cpp(program)
        elif ext == "csharp":
            cls.run_csharp(program)
        elif ext == "java":
            cls.run_java(program)
        elif ext == "py":
            cls.run_python(program)
        elif ext == "js":
            cls.run_javascript(program)
        elif ext == "r":
            cls.run_r(program)
        elif ext == "go":
            cls.run_go(program)
        elif ext == "rs":
            cls.run_rust(program)
        else:
            raise ValueError(f"Unsupported language: {ext}")
