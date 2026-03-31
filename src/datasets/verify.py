import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from prettytable import PrettyTable

from src.execution.tester import Tester
from src.utils import Loader

class DatasetVerifier:
    @classmethod
    def run(cls, problem: str | None = None):
        paths = sorted(glob.glob(os.path.join("data", "*", "dataset.json")))
        if not paths:
            print(f"No dataset.json files found under data")
            return {}

        if problem:
            safe = problem.replace("/", "_")
            paths = [path for path in paths if Path(path).parts[-2] == safe]
            if not paths:
                print(f"No dataset.json found for problem {problem}")
                return {}

        total = matched = 0
        pct = lambda m, t: f"{m/t*100:.1f}%" if t else "n/a"
        per_verdict = defaultdict(lambda: {"match": 0, "total": 0})
        per_ext = defaultdict(lambda: {"match": 0, "total": 0})

        loader = Loader()
        for path in paths:
            assignment, timelimit, memlimit, buggys, references, testcases = \
                loader.run(path)
            Tester.init_globals(testcases, timelimit, memlimit)
            tot = len(buggys)+len(references)
            pbar = tqdm(total=tot, desc=assignment['id'])
            mismatches = []
            stored = "failed"
            for bug in buggys:
                ext = bug.ext
                results = Tester.run(bug)
                passed = Tester.is_all_pass(results)

                total += 1
                per_verdict[stored]["total"] += 1
                per_ext[ext]["total"] += 1

                if not passed:
                    matched += 1
                    per_verdict[stored]["match"] += 1
                    per_ext[ext]["match"] += 1
                else:
                    mismatches.append(bug.id)
                pbar.update(1)
                pbar.set_postfix({"match": f"{pct(tot-len(mismatches), tot)}"})

            stored = "passed"
            for ref in references:
                ext = ref.ext
                results = Tester.run(ref)
                passed = Tester.is_all_pass(results)

                total += 1
                per_verdict[stored]["total"] += 1
                per_ext[ext]["total"] += 1

                if passed:
                    matched += 1
                    per_verdict[stored]["match"] += 1
                    per_ext[ext]["match"] += 1
                else:
                    mismatches.append(ref.id)
                pbar.update(1)
                pbar.set_postfix({"match": f"{pct(tot-len(mismatches), tot)}"})
            pbar.close()
            
            with open(path, 'r') as f:
                dataset = json.load(f)
            existing = dataset.get('mismatches', [])
            dataset['mismatches'] = list(set(existing) | set(mismatches))
            with open(path, 'w') as f:
                json.dump(dataset, f, indent=4)

        overview = PrettyTable(["Metric", "Value"])
        overview.align["Metric"] = "l"
        overview.align["Value"] = "r"
        overview.add_row(["Problems", len(paths)])
        overview.add_row(["Submissions", total])
        overview.add_row(["Matched", f"{matched} ({pct(matched, total)})"])
        overview.add_row(["Mismatched", f"{total - matched} ({pct(total - matched, total)})"])
        print(overview)

        verdict_table = PrettyTable(["Verdict", "Total", "Match", "Accuracy"])
        verdict_table.align["Verdict"] = "l"
        verdict_table.align["Total"] = "r"
        verdict_table.align["Match"] = "r"
        verdict_table.align["Accuracy"] = "r"
        for verdict in sorted(per_verdict):
            match_count = per_verdict[verdict]["match"]
            total_count = per_verdict[verdict]["total"]
            verdict_table.add_row([verdict, total_count, match_count, pct(match_count, total_count)])
        print(verdict_table)

        ext_table = PrettyTable(["Ext", "Total", "Match", "Accuracy"])
        ext_table.align["Ext"] = "l"
        ext_table.align["Total"] = "r"
        ext_table.align["Match"] = "r"
        ext_table.align["Accuracy"] = "r"
        for ext in sorted(per_ext):
            match_count = per_ext[ext]["match"]
            total_count = per_ext[ext]["total"]
            ext_table.add_row([ext, total_count, match_count, pct(match_count, total_count)])
        print(ext_table)

        return {
            "problems": len(paths),
            "submissions": total,
            "matched": matched,
            "mismatched": total - matched,
            "per_verdict": dict(per_verdict),
            "per_ext": dict(per_ext),
        }
