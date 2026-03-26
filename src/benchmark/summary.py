"""Print a summary of the locally stored benchmark datasets."""

import json
import os
from collections import defaultdict

from prettytable import PrettyTable


class BenchmarkSummary:
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
        verdict_counts: dict[str, int] = defaultdict(int)
        lang_counts: dict[str, int] = defaultdict(int)
        problem_rows: list[dict] = []

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
            assignment = dataset.get("assignment", {})

            sub_count = len(subs)
            total_subs += sub_count

            v_counts: dict[str, int] = defaultdict(int)
            l_counts: dict[str, int] = defaultdict(int)
            total_loc = 0
            for s in subs:
                v_counts[s["status"]] += 1
                l_counts[s["ext"]] += 1
                verdict_counts[s["status"]] += 1
                lang_counts[s["ext"]] += 1
                total_loc += s.get("code", "").count("\n") + 1

            avg_loc = total_loc / sub_count if sub_count else 0

            problem_rows.append({
                "id": assignment.get("id", problem_dir),
                "title": assignment.get("title", ""),
                "subs": sub_count,
                "tests": len(tests),
                "avg_loc": avg_loc,
                "verdicts": dict(v_counts),
                "langs": dict(l_counts),
            })

        # ── Overview ──
        print(f"\n{'─' * 60}")
        overview = PrettyTable(["Metric", "Value"])
        overview.align["Metric"] = "l"
        overview.align["Value"] = "r"
        overview.add_row(["Problems", f"{len(problem_rows):,}"])
        overview.add_row(["Total submissions", f"{total_subs:,}"])
        overview.add_row(["Avg submissions/problem", f"{total_subs / len(problem_rows):,.1f}" if problem_rows else "0"])
        print(overview)

        # ── Per-problem ──
        verdict_keys = ["OK", "WRONG_ANSWER", "TIME_LIMIT_EXCEEDED", "MEMORY_LIMIT_EXCEEDED"]
        verdict_labels = ["OK", "WA", "TLE", "MLE"]
        fields = ["Problem", "Tests", "Avg. LOC", "Subs"] + verdict_labels
        pt = PrettyTable(fields)
        pt.align["Problem"] = "l"
        for k in ["Tests", "Avg. LOC", "Subs"] + verdict_labels:
            pt.align[k] = "r"

        def _fmt(count: int, total: int) -> str:
            pct = count / total * 100 if total else 0
            return f"{count:,} ({pct:.1f}%)"

        all_avg_locs = [r["avg_loc"] for r in problem_rows]

        for row in problem_rows:
            pt.add_row([
                row["id"],
                row["tests"],
                f"{row['avg_loc']:.1f}",
                f"{row['subs']:,}",
            ] + [
                _fmt(row["verdicts"].get(v, 0), row["subs"]) for v in verdict_keys
            ])

        total_avg_loc = sum(all_avg_locs) / len(all_avg_locs) if all_avg_locs else 0
        pt.add_row([""] * len(fields))
        pt.add_row([
            "TOTAL", "",
            f"{total_avg_loc:.1f}",
            f"{total_subs:,}",
        ] + [
            _fmt(verdict_counts.get(v, 0), total_subs) for v in verdict_keys
        ])

        print(pt)

        # ── Verdict distribution ──
        vt = PrettyTable(["Verdict", "Count", "%"])
        vt.align["Verdict"] = "l"
        vt.align["Count"] = "r"
        vt.align["%"] = "r"
        for v in sorted(verdict_counts, key=verdict_counts.get, reverse=True):
            pct = verdict_counts[v] / total_subs * 100 if total_subs else 0
            vt.add_row([v, f"{verdict_counts[v]:,}", f"{pct:.1f}%"])
        print(vt)

        # ── Language distribution ──
        lt = PrettyTable(["Language", "Count", "%"])
        lt.align["Language"] = "l"
        lt.align["Count"] = "r"
        lt.align["%"] = "r"
        for ext in sorted(lang_counts, key=lang_counts.get, reverse=True):
            pct = lang_counts[ext] / total_subs * 100 if total_subs else 0
            lt.add_row([ext, f"{lang_counts[ext]:,}", f"{pct:.1f}%"])
        print(lt)
