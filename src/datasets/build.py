import json
import os
from collections import defaultdict

from datasets import load_dataset
from dotenv import load_dotenv
from prettytable import PrettyTable
from tqdm import tqdm

load_dotenv()


class DatasetBuilder:
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
        subs_ds = load_dataset("open-r1/codeforces-submissions", split="train")
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
    def write_dataset(
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
        return cls.write_dataset(groups, problems, min_count)
