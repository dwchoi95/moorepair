import os
import csv
import json
import pickle
import time
import math
import pandas as pd
from tqdm import tqdm
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

from src.llms import Spec
from src.genetic import GA, Fitness, Selection
from src.utils import ETC, TED, Loader
from src.execution import Tester, Programs, TestCases


class Experiments:
    def __init__(self,
        generations:int=3, pop_size:int=10, initialization:bool=False,
        selection:str="nsga3", threshold:float=0.5,
        llm:str="codellama:7b", temperature:float=0.8, timelimit:int=1,
        objectives:list=Fitness.OBJECTIVES, trials:int=10,
        sampling:bool=False, reset:bool=False, multi:bool=False
    ):  
        self.loader = Loader(sampling, initialization)
        
        self.generations = generations
        self.pop_size = pop_size
        self.selection = selection
        self.threshold = threshold
        self.timelimit = timelimit
        self.fitness = Fitness(objectives)
        self.select = Selection(self.fitness)
        Spec.set(llm, temperature)
        self.trials = trials
        self.reset = reset
        self.multi = multi
        
        self.obj = "".join(objectives)
    
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
        generation_stats = {}
        table = PrettyTable([
            "#Trial",
            "#Generation",
            "#References",
            "#Buggys",
            "#Fixed",
            "%Repair Rate",
            "%Similarity",
            "%Execution Time",
            "%Memory Usage"
        ])
        
        for b_id, result in tqdm(results.items(), desc="Save", leave=False):
            # No solution found
            if not result: continue
            
            buggy = buggys.get_prog_by_id(b_id)
            refer = references.get_prog_by_id(b_id)
            
            # Generation별로 처리
            for gen, solutions in result.items():
                if gen not in generation_stats:
                    generation_stats[gen] = {
                        'accuracy': 0,
                        'similarity': 0,
                        'runtime': 0,
                        'memory': 0,
                        'count': 0,
                        'num_solutions': 0
                    }
                
                generation_stats[gen]['num_solutions'] += len(solutions)
            
                # Selection of best solution for this generation
                scoring = {}
                for patch in solutions:
                    scores = self.fitness.run(buggy, patch)
                    scoring[patch.id] = scores
                if not scoring: continue
                sol_id = self.select.hype(scoring)
                patch = solutions.get_prog_by_id(sol_id)
                # Evaluation
                ## accuracy
                generation_stats[gen]['accuracy'] += 1
                
                ## similarity
                ted = TED(buggy.ext)
                refer_sim = ted.compute_levenshtein_led(buggy.code, refer.code)
                patch_sim = ted.compute_levenshtein_led(buggy.code, patch.code)
                generation_stats[gen]['similarity'] += self.ratio(
                    (refer_sim - patch_sim), (refer_sim + patch_sim))
                
                ## efficiency
                refer_time = refer.results.exec_time()
                patch_time = patch.results.exec_time()
                generation_stats[gen]['runtime'] += self.ratio(
                    (refer_time - patch_time), (refer_time + patch_time))
                
                refer_mem = refer.results.mem_usage()
                patch_mem = patch.results.mem_usage()
                generation_stats[gen]['memory'] += self.ratio(
                    (refer_mem - patch_mem), (refer_mem + patch_mem))
                
                generation_stats[gen]['count'] += 1
                
                # 마지막 generation의 best solution만 final에 저장
                if gen == max(result.keys()):
                    final.append((buggy, refer, patch))
                         
        
        total_bugs = len(buggys)
        total_refs = len(references)
        for gen in sorted(generation_stats.keys()):
            stats = generation_stats[gen]
            count = stats['count']
            
            if count > 0:
                table.add_row([
                    trial,
                    gen,
                    total_refs,
                    total_bugs,
                    count,
                    f"{(stats['accuracy'] / total_bugs * 100):.2f}%",
                    f"{(stats['similarity'] / count * 100):.2f}%",
                    f"{(stats['runtime'] / count * 100):.2f}%",
                    f"{(stats['memory'] / count * 100):.2f}%"
                ])
            else:
                table.add_row([
                    trial,
                    gen,
                    total_refs,
                    total_bugs,
                    0,
                    "0.00%",
                    "0.00%",
                    "0.00%",
                    "0.00%"
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
        df.to_csv(best_sol_path, index=False)

    def __save_experiments(self, problemId: int) -> None:
        # per-problem summary path
        results_dir = os.path.join('results', str(problemId), self.selection)
        summary_path = os.path.join(results_dir, 'summary.csv')
        if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
            return  # nothing to aggregate

        df = pd.read_csv(summary_path)

        # 안전: 컬럼명이 다르면 바로 종료(또는 raise)
        required_cols = [
            "#Trial", "#Generation", "#References", "#Buggys", "#Fixed",
            "%Repair Rate", "%Similarity", "%Execution Time", "%Memory Usage"
        ]
        for c in required_cols:
            if c not in df.columns:
                return

        # 최종 generation만 사용
        final_gen = int(df["#Generation"].max())
        dff = df[df["#Generation"] == final_gen].copy()
        if dff.empty:
            return

        # 퍼센트 문자열 -> float 변환
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

        pct_cols = ["%Repair Rate", "%Similarity", "%Execution Time", "%Memory Usage"]
        for c in pct_cols:
            dff[c] = dff[c].apply(pct_to_float)

        # 평균 계산 (final generation에서 trial별 row 평균)
        # (#Fixed도 평균 내서 "평균적으로 몇 개 고쳤나"를 보여줌)
        mean_fixed = float(dff["#Fixed"].mean()) if len(dff) else 0.0
        mean_acc = float(dff["%Repair Rate"].mean()) if len(dff) else 0.0
        mean_sim = float(dff["%Similarity"].mean()) if len(dff) else 0.0
        mean_time = float(dff["%Execution Time"].mean()) if len(dff) else 0.0
        mean_mem = float(dff["%Memory Usage"].mean()) if len(dff) else 0.0

        # references/buggys는 고정값이라 첫 row 사용
        total_refs = int(dff["#References"].iloc[0])
        total_bugs = int(dff["#Buggys"].iloc[0])

        # overall.csv에 누적 저장 (problem별 1 row)
        overall_path = os.path.join("results", "overall.csv")
        os.makedirs(os.path.dirname(overall_path), exist_ok=True)

        headers = [
            "ProblemID",
            "Selection",
            "#Trials",
            "#FinalGeneration",
            "#References",
            "#Buggys",
            "#Fixed(avg)",
            "%Repair Rate(avg)",
            "%Similarity(avg)",
            "%Execution Time(avg)",
            "%Memory Usage(avg)",
        ]

        row = [
            problemId,
            self.selection,
            int(dff["#Trial"].nunique()),
            final_gen,
            total_refs,
            total_bugs,
            f"{mean_fixed:.2f}",
            f"{mean_acc:.2f}%",
            f"{mean_sim:.2f}%",
            f"{mean_time:.2f}%",
            f"{mean_mem:.2f}%",
        ]

        file_exists = os.path.exists(overall_path) and os.path.getsize(overall_path) > 0
        with open(overall_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
        

    def __core(self, trial:int, problemId:int, description:str,
               buggys:Programs, references:Programs, testcases:TestCases):
        # Generate Feedback
        Tester.init_globals(testcases, self.timelimit)
        ga = GA(buggys, references, description, self.fitness)
        # Run MooRepair
        results = ga.run(self.generations, self.pop_size, 
                            self.selection, self.threshold)
        # Save Results
        self.__save_results(
            trial, problemId, buggys, references, results
        )
            
    def run(self, problems:list):
        for problem in problems:
            problemId, description, buggys, \
                references, testcases = self.loader.run(problem)
            for trial in range(1, self.trials+1):
                self.__core(
                    trial, problemId, description,
                    buggys, references, testcases
                )
            self.__save_experiments(problemId)