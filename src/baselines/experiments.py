import os
import csv
import shutil
import pickle
import time
import asyncio
import pandas as pd
from tqdm import tqdm
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

from .moorepair import MooRepair
from .par import PaR
from .effilearner import EffiLearner
from src.llms import Models, Tokenizer
from src.genetic import Fitness, Selection
from src.utils import ETC, Loader
from src.execution import Tester, Programs


class Experiments:
    def __init__(self,
        generations:int=5, pop_size:int=10,
        llm:str="codellama/CodeLlama-7b-Instruct-hf",
        temperature:float=0.8, sampling:bool=False, 
        approach:str="moorepair", reset:bool=False,
    ):
        self.loader = Loader(sampling)

        self.generations = generations
        self.pop_size = pop_size
        Models.set(model=llm, temperature=temperature)
        Tokenizer.set(llm)
        self.reset = reset
        self.approach = approach

        self.results_cols = [
            "Approach", "#Gen", "#Buggy", "#Fixed",
            "%RR", "%f_fail", "%f_time", "%f_mem",
        ]
        self.experiments_cols = ["ProblemID"] + self.results_cols

    # ---------------------------------------------------------------- #
    # Save results                                                     #
    # ---------------------------------------------------------------- #

    def __save_results(self, problemId:int, buggys:Programs, results:dict) -> None:
        """results: {buggy_id: {gen: [Program, ...], ...}, ...}"""

        results_dir = os.path.join('results', str(problemId))
        raw_path = os.path.join(results_dir, f'{self.approach}.pkl')
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, 'wb') as f:
            pickle.dump(results, f)

        generation_stats = {
            gen: {'fixed': 0, 'f_fail': 0.0, 'f_time': 0.0, 'f_mem': 0.0}
            for gen in range(1, self.generations + 1)
        }
        final_solutions = []
        table = PrettyTable(self.results_cols)

        for b_id, gen_result in tqdm(results.items(), desc="Save", leave=False):
            if not gen_result:
                continue
            buggy = buggys.get_prog_by_id(b_id)

            for gen in range(1, self.generations + 1):
                patches = gen_result.get(gen, [])
                if not patches:
                    continue
                patch = Selection.prioritization(patches)
                if patch.fitness["f_fail"] > 0:
                    continue

                generation_stats[gen]['fixed'] += 1

                # Buggy baseline fitness
                if buggy.fitness is None:
                    Tester.run(buggy)
                    Fitness.evaluate(buggy)
                b_fail = buggy.fitness["f_fail"]
                b_time = buggy.fitness["f_time"]
                b_mem  = buggy.fitness["f_mem"]
                p_time = patch.fitness["f_time"]
                p_mem  = patch.fitness["f_mem"]

                generation_stats[gen]['f_fail'] += patch.fitness["f_fail"]
                generation_stats[gen]['f_time'] += ETC.divide(b_time - p_time, b_time + p_time)
                generation_stats[gen]['f_mem']  += ETC.divide(b_mem  - p_mem,  b_mem  + p_mem)

                if gen == max(gen_result.keys()):
                    final_solutions.append((buggy, patch))

        total_bugs = len(buggys)
        for gen in sorted(generation_stats.keys()):
            stats = generation_stats[gen]
            fixed = stats['fixed']
            table.add_row([
                self.approach,
                gen,
                total_bugs,
                fixed,
                f"{ETC.divide(fixed, total_bugs) * 100:.2f}%",
                f"{ETC.divide(stats['f_fail'], max(fixed, 1)):.4f}",
                f"{ETC.divide(stats['f_time'], max(fixed, 1)) * 100:.2f}%",
                f"{ETC.divide(stats['f_mem'],  max(fixed, 1)) * 100:.2f}%",
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

        def pct_to_float(x):
            s = str(x).strip().rstrip("%")
            try:
                return float(s)
            except ValueError:
                return 0.0

        mean_rr       = dff["%RR"].apply(pct_to_float).mean()
        mean_f_time   = dff["%f_time"].apply(pct_to_float).mean()
        mean_f_mem    = dff["%f_mem"].apply(pct_to_float).mean()
        total_bugs    = int(dff["#Buggy"].iloc[0])
        mean_fixed    = float(dff["#Fixed"].mean())

        overall_path = os.path.join("results", f"overall.csv")
        os.makedirs(os.path.dirname(overall_path), exist_ok=True)
        row = [
            problemId,
            final_gen,
            total_bugs,
            f"{mean_fixed:.2f}",
            f"{mean_rr:.2f}%",
            f"{mean_f_time:.2f}%",
            f"{mean_f_mem:.2f}%",
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
            moo_repair = MooRepair(buggys, references, description, log_path)
            results = moo_repair.run(self.generations, self.pop_size)
        else:
            generations = (self.generations+1) * self.pop_size // 2
            par = PaR(buggys, references, description, log_path)
            corrects = asyncio.run(par.run(generations))
            effi_learner = EffiLearner(corrects, description, log_path)
            results = asyncio.run(effi_learner.run(generations))
            
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
            results_dir = os.path.join('results', str(problemId))
            if os.path.exists(results_dir) and self.reset:
                shutil.rmtree(results_dir)
            self.__core(problem)
            self.__save_experiments(problemId)
