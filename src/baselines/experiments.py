import os
import csv
import shutil
import pickle
import pandas as pd
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


class Experiments:
    def __init__(self,
        generations:int=5, pop_size:int=5, selection:bool=False,
        llm:str="codellama/CodeLlama-7b-Instruct-hf",
        temperature:float=0.8, sampling:bool=False, 
        approach:str="moorepair", reset:bool=False,
    ):
        self.loader = Loader(sampling)

        self.generations = generations
        self.pop_size = pop_size
        self.selection = selection
        Models.set(model=llm, temperature=temperature)
        Tokenizer.set(llm)
        self.reset = reset
        self.approach = approach

        self.results_cols = [
            "Approach", "#Gen", "#Buggy", "#Fixed",
            "%RR", "ET(s)", "MU(MB)", "TMU(MB*s)",
            "ΔET(%)", "ΔMU(%)", "ΔTMU(%)",
        ]
        self.experiments_cols = ["ProblemID"] + self.results_cols

    # ---------------------------------------------------------------- #
    # Save results                                                     #
    # ---------------------------------------------------------------- #

    def __save_results(self, problemId:int, buggys:Programs, results:dict):
        """results: {buggy_id: {gen: [Program, ...], ...}, ...}"""

        results_dir = os.path.join('results', str(problemId))
        raw_path = os.path.join(results_dir, f'{self.approach}.pkl')
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, 'wb') as f:
            pickle.dump(results, f)

        generation_stats = {
            gen: {
                'fixed': 0,
                'ET': 0.0, 'MU': 0.0, 'TMU': 0.0,
                'dET': 0.0, 'dMU': 0.0, 'dTMU': 0.0, 'delta_n': 0,
            }
            for gen in range(1, self.generations + 1)
        }
        final_solutions = []
        table = PrettyTable(self.results_cols)

        for b_id, gen_result in tqdm(results.items(), desc="Save", leave=False):
            if not gen_result:
                continue
            buggy = buggys.get_prog_by_id(b_id)

            # Measure buggy baseline (always valid — buggy runs but fails tests)
            buggy_results = Tester.run(buggy)
            buggy_et  = buggy_results.ET()
            buggy_mu  = buggy_results.MU()
            buggy_tmu = buggy_results.TMU()

            for gen in range(1, self.generations + 1):
                patches = gen_result.get(gen, [])
                if not patches:
                    continue
                patch = Selection.prioritization(patches)

                patch_results = Tester.run(patch, profile=True)
                generation_stats[gen]['fixed'] += 1
                generation_stats[gen]['ET']  += patch_results.ET()
                generation_stats[gen]['MU']  += patch_results.MU()
                generation_stats[gen]['TMU'] += patch_results.TMU()

                generation_stats[gen]['dET']  += ETC.divide(buggy_et  - patch_results.ET(),  buggy_et)  * 100
                generation_stats[gen]['dMU']  += ETC.divide(buggy_mu  - patch_results.MU(),  buggy_mu)  * 100
                generation_stats[gen]['dTMU'] += ETC.divide(buggy_tmu - patch_results.TMU(), buggy_tmu) * 100
                generation_stats[gen]['delta_n'] += 1

                if gen == max(gen_result.keys()):
                    final_solutions.append((buggy, patch))

        total_bugs = len(buggys)
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
        summary_path = os.path.join(results_dir, f'{self.approach}_summary.csv')
        file_exists = os.path.exists(summary_path) and os.path.getsize(summary_path) > 0
        with open(summary_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(table.field_names)
            writer.writerows(table._rows)

        # Save final solutions
        df = pd.DataFrame(columns=['ID', 'buggy', 'patch'])
        for buggy, patch in final_solutions:
            df = pd.concat([df, pd.DataFrame({
                'ID': [buggy.id],
                'buggy': [buggy.code],
                'patch': [patch.code],
            })], ignore_index=True)
        sol_path = os.path.join(results_dir, f'{self.approach}_solutions.csv')
        os.makedirs(os.path.dirname(sol_path), exist_ok=True)
        df.to_csv(sol_path, index=False)

    # ---------------------------------------------------------------- #
    # Aggregate to overall.csv                                          #
    # ---------------------------------------------------------------- #

    def __save_experiments(self, problemId:int) -> None:
        results_dir = os.path.join('results', str(problemId))
        summary_path = os.path.join(results_dir, f'{self.approach}_summary.csv')
        if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
            return

        df = pd.read_csv(summary_path)
        final_gen = int(df["#Gen"].max())
        dff = df[df["#Gen"] == final_gen].copy()
        if dff.empty:
            return

        def to_float(x):
            try:
                return float(str(x).strip().rstrip("%"))
            except ValueError:
                return 0.0

        mean_rr  = dff["%RR"].apply(to_float).mean()
        mean_et  = dff["ET(s)"].apply(to_float).mean()
        mean_mu  = dff["MU(MB)"].apply(to_float).mean()
        mean_tmu = dff["TMU(MB*s)"].apply(to_float).mean()
        total_bugs = int(dff["#Buggy"].iloc[0])
        mean_fixed = float(dff["#Fixed"].mean())

        def mean_delta(col):
            valid = dff[col][dff[col] != "N/A"].apply(to_float)
            return f"{valid.mean():.2f}%" if not valid.empty else "N/A"

        mean_det  = mean_delta("ΔET(%)")
        mean_dmu  = mean_delta("ΔMU(%)")
        mean_dtmu = mean_delta("ΔTMU(%)")

        overall_path = os.path.join("results", f"overall.csv")
        os.makedirs(os.path.dirname(overall_path), exist_ok=True)
        row = [
            problemId,
            final_gen,
            total_bugs,
            f"{mean_fixed:.2f}",
            f"{mean_rr:.2f}%",
            f"{mean_et:.4f}",
            f"{mean_mu:.4f}",
            f"{mean_tmu:.4f}",
            mean_det,
            mean_dmu,
            mean_dtmu,
        ]
        file_exists = os.path.exists(overall_path) and os.path.getsize(overall_path) > 0
        with open(overall_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.experiments_cols)
            writer.writerow(row)

    # ---------------------------------------------------------------- #
    # Core logic                                                       #
    # ---------------------------------------------------------------- #

    def __core(self, problem:str):
        problemId, description, \
            timelimit, memlimit, \
            buggys, references, testcases = \
                self.loader.run(problem)
        
        raw_path = os.path.join('results', problemId, f'{self.approach}.pkl')
        if os.path.exists(raw_path):
            with open(raw_path, 'rb') as f:
                results = pickle.load(f)
            self.__save_results(problemId, buggys, results)
            return

        log_path = os.path.join('logs', problemId, f'{self.approach}.log')
        Tester.init_globals(testcases, timelimit, memlimit)
        if self.approach == "moorepair":
            moo_repair = MooRepair(buggys, references, description, self.selection, log_path)
            results = moo_repair.run(self.generations, self.pop_size)
        else:
            parel = PaREffiLearner(buggys, references, description, log_path)
            results = parel.run(self.generations, self.pop_size)
            
        self.__save_results(problemId, buggys, results)

    # ---------------------------------------------------------------- #
    # Public entry point                                                 #
    # ---------------------------------------------------------------- #

    def run(self, problems:list) -> None:
        overall_path = os.path.join('results', 'overall.csv')
        if os.path.isfile(overall_path) and self.reset:
            os.remove(overall_path)
        for problem in problems:
            problemId = os.path.dirname(problem).split(os.sep)[-1]
            print(f"\n=== {problemId} ===")
            results_dir = os.path.join('results', problemId)
            if os.path.exists(results_dir) and self.reset:
                shutil.rmtree(results_dir)
            self.__core(problem)
            self.__save_experiments(problemId)
