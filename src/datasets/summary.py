import json
import os
from collections import defaultdict

from prettytable import PrettyTable


class DatasetSummary:
    DATA_DIR = "data"

    @classmethod
    def run(cls) -> None:
        if not os.path.isdir(cls.DATA_DIR):
            print(f"No data directory found at '{cls.DATA_DIR}'.")
            return

        problems = sorted(os.listdir(cls.DATA_DIR))
        if not problems:
            print("No problems found.")
            return

        total_subs = 0
        total_tests = 0
        total_loc = 0
        verdict_counts: dict[str, int] = defaultdict(int)
        lang_counts: dict[str, int] = defaultdict(int)
        n_problems = 0

        for problem_dir in problems:
            path = os.path.join(cls.DATA_DIR, problem_dir, "dataset.json")
            if not os.path.isfile(path):
                continue

            with open(path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            subs = dataset.get("submissions", [])
            mismatches = set(dataset.get("mismatches", []))
            subs = [s for s in subs if s["id"] not in mismatches]
            tests = dataset.get("test_cases", [])

            n_problems += 1
            total_subs += len(subs)
            total_tests += len(tests)

            for s in subs:
                verdict_counts[s["status"]] += 1
                lang_counts[s["ext"]] += 1
                total_loc += s.get("code", "").count("\n") + 1

        avg_loc = total_loc / total_subs if total_subs else 0
        langs = ", ".join(
            ext for ext in sorted(lang_counts, key=lang_counts.get, reverse=True)
        )

        verdict_keys = ["OK", "WRONG_ANSWER", "TIME_LIMIT_EXCEEDED", "MEMORY_LIMIT_EXCEEDED"]
        verdict_labels = ["OK", "WA", "TLE", "MLE"]

        table = PrettyTable(["Statistic", "Value"])
        table.align["Statistic"] = "l"
        table.align["Value"] = "r"

        table.add_row(["Problems", f"{n_problems:,}"])
        table.add_row(["Test cases", f"{total_tests:,}"])
        table.add_row(["Submissions", f"{total_subs:,}"])
        table.add_row(["Avg. LOC", f"{avg_loc:.1f}"])
        for vk, vl in zip(verdict_keys, verdict_labels):
            c = verdict_counts.get(vk, 0)
            pct = c / total_subs * 100 if total_subs else 0
            table.add_row([f"Verdict({vl})", f"{c:,} ({pct:.1f}%)"])
        table.add_row(["Language", langs])

        print(table)
