import os
import csv
from tqdm import tqdm
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

from .moorepair import MooRepair
from .parel import PaREffiLearner
from src.llms import Models, Tokenizer
from src.genetic import Selection
from src.utils import ETC, Loader
from src.execution import Tester, Programs


OVERALL_PATH = "overall.csv"
OVERALL_COLS = [
    "ProblemID", "LLM", "Approach", "#Gen",
    "Verdict", "#Buggy",  "#Fixed", "%RR",
    "RPS", "ATT(s)",
    "ET(s)", "MU(MB)", "TMU(MB*s)",
    "ΔET(%)", "ΔMU(%)", "ΔTMU(%)",
]


class Experiments:
    def __init__(self,
        approach:str="moorepair", sampling:bool=False,
        generations:int=4, pop_size:int=6,
        llm:str="gpt-3.5-turbo", temperature:float=0.8, timeout:int=60,
        reset:bool=False, ablation:str=None, api_base:str=None
    ):
        self.loader = Loader(sampling)

        self.approach = approach
        self.ablation = ablation
        # Approach label recorded in overall.csv
        self.label = f"{approach}[{ablation}]" if ablation else approach
        self.generations = generations
        self.pop_size = pop_size
        self.llm = llm
        Models.set(model=llm, temperature=temperature, timeout=timeout,
                   api_base=api_base)
        if not llm.startswith("gpt-"):
            Tokenizer.set(llm)
        if reset and os.path.exists(OVERALL_PATH):
            os.remove(OVERALL_PATH)

    def __save(self, problemId: str, buggys: Programs, results: dict,
               times: dict = None, label: str = None):
        """Compute per-verdict stats and append to overall.csv."""

        N = self.generations + 1
        times = times or {}
        label = label or self.label

        # Group buggys by verdict
        verdict_buggys = {}  # verdict -> [buggy_id, ...]
        for buggy in buggys:
            v = buggy.meta.get("verdict", "UNKNOWN")
            verdict_buggys.setdefault(v, []).append(buggy.id)

        # Per (gen, verdict) stats
        keys = [(gen, v) for gen in range(1, N) for v in sorted(verdict_buggys.keys())]
        stats = {
            k: {'total': len(verdict_buggys[k[1]]), 'fixed': 0,
                 'ET': 0.0, 'MU': 0.0, 'TMU': 0.0,
                 'dET': 0.0, 'dMU': 0.0, 'dTMU': 0.0, 'delta_n': 0,
                 'RPS': 0.0, 'ATT': 0.0, 'att_n': 0}
            for k in keys
        }

        for b_id, gen_result in tqdm(results.items(), desc="Save", leave=False):
            buggy = buggys.get_prog_by_id(b_id)
            v = buggy.meta.get("verdict", "UNKNOWN")

            # ATT: wall-clock repair time, over all attempted buggys
            b_times = times.get(b_id, {})
            for gen, elapsed in b_times.items():
                if (gen, v) in stats:
                    stats[(gen, v)]['ATT'] += elapsed
                    stats[(gen, v)]['att_n'] += 1

            if not gen_result:
                continue

            buggy_results = Tester.run(buggy)
            buggy_et  = buggy_results.ET()
            buggy_mu  = buggy_results.MU()
            buggy_tmu = buggy_results.TMU()

            for gen in range(1, N):
                patches = gen_result.get(gen, [])
                if not patches:
                    continue
                patch = Selection.prioritization(patches)
                patch_results = Tester.run(patch, profiling=True)
                patch_et = patch_results.ET()
                patch_mu = patch_results.MU()
                patch_tmu = patch_results.TMU()

                s = stats[(gen, v)]
                s['fixed'] += 1
                s['ET']  += patch_et
                s['MU']  += patch_mu
                s['TMU'] += patch_tmu
                s['dET']  += ETC.divide(buggy_et - patch_et, buggy_et + patch_et) * 100
                s['dMU']  += ETC.divide(buggy_mu - patch_mu, buggy_mu + patch_mu) * 100
                s['dTMU'] += ETC.divide(buggy_tmu - patch_tmu, buggy_tmu + patch_tmu) * 100
                s['delta_n'] += 1
                s['RPS'] += ETC.rps(buggy.code, patch.code)

        # Build rows
        rows = []
        for gen, v in keys:
            s = stats[(gen, v)]
            fixed = s['fixed']
            total = s['total']
            n = max(fixed, 1)
            dn = s['delta_n']
            rows.append([
                problemId,
                self.llm,
                label,
                gen,
                v,
                total,
                fixed,
                f"{ETC.divide(fixed, total) * 100:.2f}%",
                f"{ETC.divide(s['RPS'], dn):.4f}" if dn > 0 else "N/A",
                f"{ETC.divide(s['ATT'], s['att_n']):.2f}" if s['att_n'] > 0 else "N/A",
                f"{ETC.divide(s['ET'], n):.4f}",
                f"{ETC.divide(s['MU'], n):.4f}",
                f"{ETC.divide(s['TMU'], n):.4f}",
                f"{ETC.divide(s['dET'], dn):.2f}%" if dn > 0 else "N/A",
                f"{ETC.divide(s['dMU'], dn):.2f}%" if dn > 0 else "N/A",
                f"{ETC.divide(s['dTMU'], dn):.2f}%" if dn > 0 else "N/A",
            ])

        # Print aggregated final-generation result (across all verdicts)
        last_gen = self.generations
        agg = {'total': 0, 'fixed': 0, 'ET': 0.0, 'MU': 0.0, 'TMU': 0.0,
               'dET': 0.0, 'dMU': 0.0, 'dTMU': 0.0, 'delta_n': 0,
               'RPS': 0.0, 'ATT': 0.0, 'att_n': 0}
        for v in sorted(verdict_buggys.keys()):
            s = stats[(last_gen, v)]
            agg['total']   += s['total']
            agg['fixed']   += s['fixed']
            agg['ET']      += s['ET']
            agg['MU']      += s['MU']
            agg['TMU']     += s['TMU']
            agg['dET']     += s['dET']
            agg['dMU']     += s['dMU']
            agg['dTMU']    += s['dTMU']
            agg['delta_n'] += s['delta_n']
            agg['RPS']     += s['RPS']
            agg['ATT']     += s['ATT']
            agg['att_n']   += s['att_n']
        fixed = agg['fixed']
        total = agg['total']
        n  = max(fixed, 1)
        dn = agg['delta_n']
        summary_row = [
            problemId, self.llm, label, last_gen, "ALL",
            total, fixed,
            f"{ETC.divide(fixed, total) * 100:.2f}%",
            f"{ETC.divide(agg['RPS'], dn):.4f}" if dn > 0 else "N/A",
            f"{ETC.divide(agg['ATT'], agg['att_n']):.2f}" if agg['att_n'] > 0 else "N/A",
            f"{ETC.divide(agg['ET'],  n):.4f}",
            f"{ETC.divide(agg['MU'],  n):.4f}",
            f"{ETC.divide(agg['TMU'], n):.4f}",
            f"{ETC.divide(agg['dET'],  dn):.2f}%" if dn > 0 else "N/A",
            f"{ETC.divide(agg['dMU'],  dn):.2f}%" if dn > 0 else "N/A",
            f"{ETC.divide(agg['dTMU'], dn):.2f}%" if dn > 0 else "N/A",
        ]
        # Persist the cross-verdict aggregate as Verdict="ALL" rows so
        # stats.py can run statistical tests directly on overall.csv
        rows.append(summary_row)
        table = PrettyTable(["Metric", "Value"])
        table.align["Metric"] = "r"
        table.align["Value"] = "l"
        for col, val in zip(OVERALL_COLS, summary_row):
            table.add_row([col, val])
        print(table)

        # Append to overall.csv
        file_exists = os.path.exists(OVERALL_PATH) and os.path.getsize(OVERALL_PATH) > 0
        with open(OVERALL_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(OVERALL_COLS)
            writer.writerows(rows)

    def __core(self, problem: str):
        assignment, timelimit, memlimit, buggys, references, testcases = \
            self.loader.run(problem)
        problemId = assignment['id'].replace("/", "_")
        print(f"\n=== {problemId} ===")

        Tester.init_globals(testcases, timelimit, memlimit)

        if self.approach == "PaREL":
            parel = PaREffiLearner(buggys, references, assignment)
            results = parel.run(self.generations, self.pop_size)
            # One run yields both baselines: PaR (correct patches before
            # EffiLearner) and PaREL (after EffiLearner)
            self.__save(problemId, buggys, parel.par_results,
                        parel.par_times, label="PaR")
            self.__save(problemId, buggys, results, parel.times)
        else:
            moo_repair = MooRepair(buggys, references, assignment,
                                   ablation=self.ablation)
            results = moo_repair.run(self.generations, self.pop_size)
            self.__save(problemId, buggys, results, moo_repair.times)

    def run(self, problems: list) -> None:
        for problem in problems:
            self.__core(problem)
