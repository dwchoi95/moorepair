"""
Build MooRepair-format benchmark from HuggingFace Codeforces datasets.
"""

import json
import os
from collections import defaultdict

from datasets import load_dataset
from dotenv import load_dotenv
from prettytable import PrettyTable
from tqdm import tqdm

load_dotenv()


class BenchmarkBuilder:
    KEEP_VERDICT = {"OK", "WRONG_ANSWER", "TIME_LIMIT_EXCEEDED", "MEMORY_LIMIT_EXCEEDED"}
    GENERATED_TESTS_REPO = "open-r1/codeforces"
    _generated_tests_cache: dict[str, dict[str, list[dict]]] = {}

    LANGUAGES = {
        "Python 3": "py",
        "PyPy 3": "py",
        "PyPy 3-64": "py",
        "GNU C++": "cpp",
        "GNU C++11": "cpp",
        "GNU C++17": "cpp",
        "GNU C++17 Diagnostics": "cpp",
        "C++14 (GCC 6-32)": "cpp",
        "C++17 (GCC 7-32)": "cpp",
        "C++20 (GCC 13-64)": "cpp",
        "MS C++": "cpp",
        "GNU C": "c",
        "Java 7": "java",
        "Java 8": "java",
        "Java 11": "java",
        "MS C#": "cs",
        "FPC": "pas",
        "PascalABC.NET": "pas",
        "Haskell": "hs",
    }

    @classmethod
    def load_problems(cls) -> dict:
        print("Loading codeforces problems...")
        problems_ds = load_dataset("open-r1/codeforces", name="verifiable")
        problems = {}
        for split in ["train", "test"]:
            for row in problems_ds[split]:
                problems[row["id"]] = row
        return problems

    @classmethod
    def load_submissions(cls, valid_ids: set, language: str | None) -> dict:
        print("Loading codeforces submissions...")
        subs_ds = load_dataset("open-r1/codeforces-submissions", name="default", split="train")
        subs_ds = subs_ds.filter(
            lambda row: (
                row["problem_id"] in valid_ids
                and (language is None or row["programmingLanguage"] == language)
                and row["verdict"] in cls.KEEP_VERDICT
                and row["testset"] == "TESTS"
            ),
            num_proc=os.cpu_count(),
        )
        groups = defaultdict(list)
        for row in tqdm(subs_ds, desc="Grouping", unit="row"):
            lang = row["programmingLanguage"]
            groups[row["problem_id"]].append({
                "id": str(row["submission_id"]),
                "code": str(row["source"]),
                "ext": cls.LANGUAGES.get(lang, lang),
                "status": str(row["verdict"]),
            })
        return groups

    @staticmethod
    def build_assignment(problem: dict):
        return {
            "id": problem.get("id"),
            "title": problem.get("title"),
            "description": problem.get("description"),
            "input_format": problem.get("input_format"),
            "output_format": problem.get("output_format"),
            "interaction_format": problem.get("interaction_format"),
            "note": problem.get("note"),
            "time_limit": problem.get("time_limit"),
            "memory_limit": problem.get("memory_limit")
        }

    @staticmethod
    def _base_test_cases(problem: dict) -> list[dict]:
        official = problem.get("official_tests") or []
        if official:
            return [
                {"input": tc.get("input", ""), "output": tc.get("output", "")}
                for tc in official
            ]
        examples = problem.get("examples") or []
        return [
            {"input": tc.get("input", ""), "output": tc.get("output", "")}
            for tc in examples
        ]

    @staticmethod
    def _contest_id(problem_id: str) -> str | None:
        if "/" not in problem_id:
            return None
        return problem_id.split("/", 1)[0]

    @classmethod
    def _load_generated_tests_for_contest(
        cls,
        contest_id: str,
    ) -> dict[str, list[dict]]:
        if contest_id in cls._generated_tests_cache:
            return cls._generated_tests_cache[contest_id]

        parquet_uri = None
        candidate_names = []
        if contest_id.isdigit():
            candidate_names.append(f"test_cases_{int(contest_id):04d}.parquet")
        candidate_names.append(f"test_cases_{contest_id}.parquet")
        for candidate_name in candidate_names:
            candidate_uri = f"hf://datasets/{cls.GENERATED_TESTS_REPO}/generated_tests/{candidate_name}"
            try:
                import pyarrow.parquet as pq

                pq.read_schema(candidate_uri)
                parquet_uri = candidate_uri
                break
            except Exception:
                continue

        if parquet_uri is None:
            cls._generated_tests_cache[contest_id] = {}
            return cls._generated_tests_cache[contest_id]

        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(parquet_uri).names)
        index_column = "test_case_i" if "test_case_i" in schema_names else "test_i"
        columns = ["problem_id", "input", "output", index_column]
        table = pq.read_table(parquet_uri, columns=columns)
        rows = zip(
            table.column("problem_id").to_pylist(),
            table.column("input").to_pylist(),
            table.column("output").to_pylist(),
            table.column(index_column).to_pylist(),
        )

        grouped = defaultdict(list)
        for row_problem_id, tc_input, tc_output, tc_index in rows:
            grouped[str(row_problem_id)].append({
                "input": "" if tc_input is None else str(tc_input),
                "output": "" if tc_output is None else str(tc_output),
                "test_case_i": int(tc_index),
            })

        cls._generated_tests_cache[contest_id] = {
            problem_id: sorted(testcases, key=lambda testcase: testcase["test_case_i"])
            for problem_id, testcases in grouped.items()
        }
        return cls._generated_tests_cache[contest_id]

    @classmethod
    def extract_test_cases(
        cls,
        problem: dict,
    ) -> list[dict]:
        merged = []
        seen = set()

        for testcase in cls._base_test_cases(problem):
            key = (str(testcase["input"]), str(testcase["output"]))
            if key in seen:
                continue
            seen.add(key)
            merged.append({
                "input": str(testcase["input"]),
                "output": str(testcase["output"]),
            })

        # problem_id = str(problem.get("id") or "")
        # contest_id = cls._contest_id(problem_id)
        # if contest_id:
        #     contest_tests = cls._load_generated_tests_for_contest(contest_id)
        #     for testcase in contest_tests.get(problem_id, []):
        #         key = (testcase["input"], testcase["output"])
        #         if key in seen:
        #             continue
        #         seen.add(key)
        #         merged.append({
        #             "input": testcase["input"],
        #             "output": testcase["output"],
        #         })

        return [
            {"id": index, "input": testcase["input"], "output": testcase["output"]}
            for index, testcase in enumerate(merged, start=1)
        ]

    @classmethod
    def passes_min_filter(cls, submissions: list[dict], min_count: int) -> bool:
        counts = defaultdict(int)
        for submission in submissions:
            counts[submission["status"]] += 1
        return all(counts[verdict] >= min_count for verdict in cls.KEEP_VERDICT)

    @classmethod
    def write_problem(
        cls,
        problem_id: str,
        problem: dict,
        submissions: list[dict],
        min_count: int,
    ) -> bool:
        safe_id = problem_id.replace("/", "_")
        out_path = os.path.join("data", safe_id, "dataset.json")

        if not cls.passes_min_filter(submissions, min_count):
            if os.path.exists(out_path):
                os.remove(out_path)
            return False

        test_cases = cls.extract_test_cases(problem)
        if not test_cases:
            return False

        dataset = {
            "assignment": cls.build_assignment(problem),
            "submissions": submissions,
            "test_cases": test_cases,
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        return True

    @classmethod
    def write_benchmark(
        cls,
        groups: dict,
        problems: dict,
        min_count: int,
    ) -> int:
        written = 0
        verdict_counts = defaultdict(int)
        lang_counts = defaultdict(int)
        total_subs = 0

        for problem_id, submissions in tqdm(groups.items(), desc="Writing", unit="problem"):
            problem = problems.get(problem_id)
            if problem is None:
                continue
            if cls.write_problem(
                problem_id,
                problem,
                submissions,
                min_count,
            ):
                written += 1
                total_subs += len(submissions)
                for submission in submissions:
                    verdict_counts[submission["status"]] += 1
                    lang_counts[submission["ext"]] += 1

        print(f"\n── Summary {'─' * 50}")

        overview = PrettyTable(["Metric", "Count"])
        overview.align["Metric"] = "l"
        overview.align["Count"] = "r"
        overview.add_row(["Problems", f"{written:,}"])
        overview.add_row(["Submissions", f"{total_subs:,}"])
        print(overview)

        verdict_table = PrettyTable(["Verdict", "Count", "%"])
        verdict_table.align["Verdict"] = "l"
        verdict_table.align["Count"] = "r"
        verdict_table.align["%"] = "r"
        for verdict in sorted(verdict_counts, key=verdict_counts.get, reverse=True):
            pct = verdict_counts[verdict] / total_subs * 100 if total_subs else 0
            verdict_table.add_row([verdict, f"{verdict_counts[verdict]:,}", f"{pct:.1f}%"])
        print(verdict_table)

        lang_table = PrettyTable(["Ext", "Count", "%"])
        lang_table.align["Ext"] = "l"
        lang_table.align["Count"] = "r"
        lang_table.align["%"] = "r"
        for ext in sorted(lang_counts, key=lang_counts.get, reverse=True):
            pct = lang_counts[ext] / total_subs * 100 if total_subs else 0
            lang_table.add_row([ext, f"{lang_counts[ext]:,}", f"{pct:.1f}%"])
        print(lang_table)

        return written

    @classmethod
    def run(
        cls,
        language: str | None = None,
        min_count: int = 20,
    ) -> int:
        problems = cls.load_problems()
        groups = cls.load_submissions(set(problems.keys()), language)
        return cls.write_benchmark(groups, problems, min_count)
