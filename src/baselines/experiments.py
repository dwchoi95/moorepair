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


OVERALL_PATH = os.path.join("results", "overall.csv")
OVERALL_COLS = [
    "ProblemID", "LLM", "Approach", "#Gen", 
    "#Buggy",  "#Fixed", "%RR", 
    "ET(s)", "MU(MB)", "TMU(MB*s)",
    "ΔET(%)", "ΔMU(%)", "ΔTMU(%)",
]


class Experiments:
    def __init__(self,
        generations:int=4, pop_size:int=6, selection:bool=False,
        llm:str="gpt-3.5-turbo",
        temperature:float=0.8, sampling:bool=False,
        approach:str="moorepair", reset:bool=False,
    ):
        self.loader = Loader(sampling)

        self.generations = generations
        self.pop_size = pop_size
        self.selection = selection
        self.llm = llm
        Models.set(model=llm, temperature=temperature)
        if not llm.startswith("gpt-"):
            Tokenizer.set(llm)
        self.reset = reset
        self.approach = approach

    def __save(self, problemId: str, buggys: Programs, results: dict):
        """Compute stats from results dict and append final-gen row to overall.csv."""

        N = self.generations if self.approach == "moorepair" else self.generations + 1
        generation_stats = {
            gen: {
                'fixed': 0,
                'ET': 0.0, 'MU': 0.0, 'TMU': 0.0,
                'dET': 0.0, 'dMU': 0.0, 'dTMU': 0.0, 'delta_n': 0,
            }
            for gen in range(1, N + 1)
        }

        for b_id, gen_result in tqdm(results.items(), desc="Save", leave=False):
            if not gen_result:
                continue
            buggy = buggys.get_prog_by_id(b_id)

            buggy_results = Tester.run(buggy)
            buggy_et  = buggy_results.ET()
            buggy_mu  = buggy_results.MU()
            buggy_tmu = buggy_results.TMU()

            for gen in range(1, N + 1):
                patches = gen_result.get(gen, [])
                if not patches:
                    continue
                patch = Selection.prioritization(patches)
                patch_results = Tester.run(patch, profiling=True)
                patch_et = patch_results.ET()
                patch_mu = patch_results.MU()
                patch_tmu = patch_results.TMU()

                generation_stats[gen]['fixed'] += 1
                generation_stats[gen]['ET']  += patch_et
                generation_stats[gen]['MU']  += patch_mu
                generation_stats[gen]['TMU'] += patch_tmu

                generation_stats[gen]['dET']  += ETC.divide(
                    buggy_et - patch_et,  buggy_et + patch_et) * 100
                generation_stats[gen]['dMU']  += ETC.divide(
                    buggy_mu - patch_mu,  buggy_mu + patch_mu) * 100
                generation_stats[gen]['dTMU'] += ETC.divide(
                    buggy_tmu - patch_tmu, buggy_tmu + patch_tmu) * 100
                generation_stats[gen]['delta_n'] += 1

        # Print per-gen table
        total_bugs = len(buggys)
        table = PrettyTable(OVERALL_COLS)
        for gen in sorted(generation_stats.keys()):
            stats = generation_stats[gen]
            fixed = stats['fixed']
            n = max(fixed, 1)
            dn = stats['delta_n']
            mean_et  = ETC.divide(stats['ET'],  n)
            mean_mu  = ETC.divide(stats['MU'],  n)
            mean_tmu = ETC.divide(stats['TMU'], n)
            mean_det  = f"{ETC.divide(stats['dET'],  dn):.2f}%" if dn > 0 else "N/A"
            mean_dmu  = f"{ETC.divide(stats['dMU'],  dn):.2f}%" if dn > 0 else "N/A"
            mean_dtmu = f"{ETC.divide(stats['dTMU'], dn):.2f}%" if dn > 0 else "N/A"
            table.add_row([
                problemId,
                self.llm,
                self.approach,
                gen,
                total_bugs,
                fixed,
                f"{ETC.divide(fixed, total_bugs) * 100:.2f}%",
                f"{mean_et:.4f}",
                f"{mean_mu:.4f}",
                f"{mean_tmu:.4f}",
                mean_det,
                mean_dmu,
                mean_dtmu,
            ])
        print(table)

        # Append final-gen row to overall.csv
        final_row = table._rows[-1] if table._rows else None
        if final_row:
            os.makedirs(os.path.dirname(OVERALL_PATH), exist_ok=True)
            file_exists = os.path.exists(OVERALL_PATH) and os.path.getsize(OVERALL_PATH) > 0
            with open(OVERALL_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(OVERALL_COLS)
                writer.writerow(final_row)
                
    def __core(self, problem: str):
        assignment, timelimit, memlimit, buggys, references, testcases = \
            self.loader.run(problem)
        problemId = assignment['id'].replace("/", "_")
        print(f"\n=== {problemId} ===")

        Tester.init_globals(testcases, timelimit, memlimit)

        log_path = os.path.join('logs', problemId, self.llm, f'{self.approach}.log')
        if self.approach == "moorepair":
            moo_repair = MooRepair(buggys, references, assignment, self.selection, log_path)
            results = moo_repair.run(self.generations, self.pop_size)
        else:
            parel = PaREffiLearner(buggys, references, assignment, log_path)
            results = parel.run(self.generations + 1, self.pop_size)

        self.__save(problemId, buggys, results)

    def run(self, problems: list) -> None:
        if self.reset and os.path.isfile(OVERALL_PATH):
            os.remove(OVERALL_PATH)
        for problem in problems:
            self.__core(problem)
