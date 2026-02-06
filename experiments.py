import os
import csv
import shutil
import pickle
import time
import math
import pandas as pd
from tqdm import tqdm
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

from src.llms import Models, Tokenizer
from src.genetic import GA, Fitness, Selection
from src.utils import ETC, Loader
from src.execution import Tester, Programs, TestCases


class Experiments:
    def __init__(self,
        generations:int=3, pop_size:int=10, initialization:bool=False,
        selection:str="nsga3", threshold:float=0.5,
        llm:str="codellama/CodeLlama-7b-Instruct-hf", 
        temperature:float=0.8, timelimit:int=1,
        objectives:list=Fitness.OBJECTIVES, trials:int=10,
        sampling:bool=False, reset:bool=False, multi:bool=False,
    ):  
        self.loader = Loader(sampling, initialization)
        
        self.generations = generations
        self.pop_size = pop_size
        self.selection = selection
        self.threshold = threshold
        self.timelimit = timelimit
        self.fitness = Fitness(objectives)
        self.select = Selection(self.fitness)
        Models.set(model=llm, temperature=temperature)
        Tokenizer.set(llm)
        self.trials = trials
        self.reset = reset
        self.multi = multi
        self.obj = "".join(objectives)
        
        self.evaluation_cols = [
            "%RR", "%LED", "%TED", 
            "%ET", "%MEM"
        ]
        self.results_cols = [
            "#Trial", "#Gen", "#Refer", 
            "#Buggy", "#Fixed"
        ] + self.evaluation_cols
        self.experiments_cols = [
            "ProblemID", "Selection"
        ] + self.results_cols
    
    def ratio(self, numerator: float, denominator: float) -> float:
        if numerator == 0 or denominator == 0:
            return 0.0
        value = ETC.divide(numerator, denominator)
        abs_value = abs(value)
        decimal_places = max(3, -int(math.floor(math.log10(abs_value))))
        return round(value, decimal_places)
        
    def __save_results(self, trial:int, problemId:int, 
                       buggys:Programs, references:Programs, 
                       results:dict[str, dict[int, Programs]]) -> None:
        
        # Save Results
        results_dir = os.path.join('results', str(problemId), self.selection)
        ## Save raw results
        raw_results_path = os.path.join(results_dir, 'raw', f'trial_{trial}.pkl')
        os.makedirs(os.path.dirname(raw_results_path), exist_ok=True)
        with open(raw_results_path, 'wb') as f:
            pickle.dump(results, f)
        
        final = []
        generation_stats = {gen: { 'acc': 0, 'led': 0, 'ted': 0, 'et': 0, 'mem': 0 } 
                            for gen in range(1, self.generations+1)}
        table = PrettyTable(self.results_cols)
        
        for b_id, result in tqdm(results.items(), desc="Save", leave=False):
            # No solution found
            if not result: continue
            
            buggy = buggys.get_prog_by_id(b_id)
            refer = references.get_prog_by_id(b_id)
            
            for gen in range(1, self.generations+1):
                if gen not in result: continue
                solutions = result[gen]
                if not solutions: continue
                
                # Selection of best solution in this generation
                patch = self.select.run(buggy, solutions, 1, "hype")[0]
                
                # Evaluation
                union = solutions.copy()
                union.append(refer)
                scores = self.fitness.evaluate(buggy, union)
                patch_score = scores[patch.id]
                refer_score = scores[refer.id]
                
                ## accuracy
                generation_stats[gen]['acc'] += 1
                
                ## similarity
                ### Line-level Edit Distance
                refer_led = refer_score['f3']
                patch_led = patch_score['f3']
                generation_stats[gen]['led'] += self.ratio(
                    (refer_led - patch_led), (refer_led + patch_led))
                
                ### AST-level Edit Distance
                refer_ted = refer_score['f4']
                patch_ted = patch_score['f4']
                generation_stats[gen]['ted'] += self.ratio(
                    (refer_ted - patch_ted), (refer_ted + patch_ted))
                
                ## efficiency
                ### Execution Time
                refer_time = refer_score['f5']
                patch_time = patch_score['f5']
                generation_stats[gen]['et'] += self.ratio(
                    (refer_time - patch_time), (refer_time + patch_time))
                
                ### Memory Usage
                refer_mem = refer_score['f6']
                patch_mem = patch_score['f6']
                generation_stats[gen]['mem'] += self.ratio(
                    (refer_mem - patch_mem), (refer_mem + patch_mem))
                
                if gen == max(result.keys()):
                    final.append((buggy, refer, patch))
                         
        total_bugs = len(buggys)
        total_refs = len(references)
        for gen in sorted(generation_stats.keys()):
            stats = generation_stats[gen]
            fixed = stats['acc']
            table.add_row([
                trial,
                gen,
                total_refs,
                total_bugs,
                fixed,
                f"{ETC.divide(fixed, total_bugs) * 100:.2f}%",
                f"{ETC.divide(stats['led'], fixed) * 100:.2f}%",
                f"{ETC.divide(stats['ted'], fixed) * 100:.2f}%",
                f"{ETC.divide(stats['et'], fixed) * 100:.2f}%",
                f"{ETC.divide(stats['mem'], fixed) * 100:.2f}%"
            ])
        
        print(table)
        summary_path = os.path.join(results_dir, 'summary.csv')
        headers = table.field_names
        rows = table._rows
        file_exists = os.path.exists(summary_path) and os.path.getsize(summary_path) > 0
        with open(summary_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerows(rows)
        
        # Save Final Solutions
        df = pd.DataFrame(columns=['ID', 'buggy', 'correct', 'patch'])
        for buggy, refer, patch in final:
            new_row = pd.DataFrame({
                'ID': [buggy.id],
                'buggy': [buggy.code],
                'correct': [refer.code],
                'patch': [patch.code]
            })
            df = pd.concat([df, new_row], ignore_index=True)
        best_sol_path = os.path.join(results_dir, 'solutions', f'trial_{trial}.csv')
        os.makedirs(os.path.dirname(best_sol_path), exist_ok=True)
        df.to_csv(best_sol_path, index=False, quoting=csv.QUOTE_ALL)

    def __save_experiments(self, problemId: int) -> None:
        # per-problem summary path
        results_dir = os.path.join('results', str(problemId), self.selection)
        summary_path = os.path.join(results_dir, 'summary.csv')
        if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
            return  # nothing to aggregate

        df = pd.read_csv(summary_path)

        for c in self.results_cols:
            if c not in df.columns:
                return

        # Use only the final generation for each trial
        final_gen = int(df["#Gen"].max())
        dff = df[df["#Gen"] == final_gen].copy()
        if dff.empty:
            return

        # Convert percentage strings to float
        def pct_to_float(x):
            if pd.isna(x):
                return 0.0
            s = str(x).strip()
            if s.endswith("%"):
                s = s[:-1]
            try:
                return float(s)
            except ValueError:
                return 0.0
            
        for c in self.evaluation_cols:
            dff[c] = dff[c].apply(pct_to_float)

        # Calculate averages (mean of rows per trial in the final generation)
        # (Also average #Fixed to show "average number fixed")
        total_refs = int(dff["#Refer"].iloc[0])
        total_bugs = int(dff["#Buggy"].iloc[0])
        mean_fixed = float(dff["#Fixed"].mean()) if len(dff) else 0.0
        mean_acc = float(dff["%RR"].mean()) if len(dff) else 0.0
        mean_sim_led = float(dff["%LED"].mean()) if len(dff) else 0.0
        mean_sim_ted = float(dff["%TED"].mean()) if len(dff) else 0.0
        mean_time = float(dff["%ET"].mean()) if len(dff) else 0.0
        mean_mem = float(dff["%MEM"].mean()) if len(dff) else 0.0

        # Save to overall results
        overall_path = os.path.join("results", "overall.csv")
        os.makedirs(os.path.dirname(overall_path), exist_ok=True)

        row = [
            problemId,
            self.selection,
            int(dff["#Trial"].nunique()),
            final_gen,
            total_refs,
            total_bugs,
            f"{mean_fixed:.2f}",
            f"{mean_acc:.2f}%",
            f"{mean_sim_led:.2f}%",
            f"{mean_sim_ted:.2f}%",
            f"{mean_time:.2f}%",
            f"{mean_mem:.2f}%",
        ]

        file_exists = os.path.exists(overall_path) and os.path.getsize(overall_path) > 0
        with open(overall_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.experiments_cols)
            writer.writerow(row)
        

    def __core(self, trial:int, problemId:int, description:str,
               buggys:Programs, references:Programs, testcases:TestCases):
        # Generate Feedback
        log_path = os.path.join('logs', str(problemId), self.selection, f'trial_{trial}.log')
        Tester.init_globals(testcases, self.timelimit)
        ga = GA(buggys, references, description, self.fitness, log_path)
        # Run MooRepair
        results = ga.run(self.generations, self.pop_size, 
                            self.selection, self.threshold)
        # Save Results
        self.__save_results(
            trial, problemId, buggys, references, results
        )
            
    def run(self, problems:list):
        experiments_path = os.path.join('results', 'overall.csv')
        if os.path.isfile(experiments_path) and self.reset:
            os.remove(experiments_path)
        for problem in problems:
            problemId, description, buggys, \
                references, testcases = self.loader.run(problem)
            print(f"\n=== {problemId} ===")
            results_dir = os.path.join('results', str(problemId), self.selection)
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
            for trial in range(1, self.trials+1):
                self.__core(
                    trial, problemId, description,
                    buggys, references, testcases
                )
            self.__save_experiments(problemId)