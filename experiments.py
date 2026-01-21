import json
import time
import math
import dataset
import pandas as pd
from tqdm import tqdm
from texttable import Texttable
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

from src.genetic import GA, Fitness, Selection
from src.utils import ETC, TED, Setup
from src.execution import Tester, Programs


class Experiments:
    def __init__(self,
        generations:int=3, pop_size:int=10, initialization:bool=False,
        selection:str="nsga3", threshold:float=0.5,
        llm:str="gpt-3.5-turbo", temperature:float=0.8, timelimit:int=1,
        objectives:list=Fitness.OBJECTIVES, trials:int=10,
        sampling:bool=False, reset:bool=False, multi:bool=False
    ):
        self.setup = Setup(sampling, initialization)
        
        self.generations = generations
        self.pop_size = pop_size
        self.selection = selection
        self.threshold = threshold
        self.timelimit = timelimit
        self.objectives = objectives
        self.llm = llm
        self.temperature = temperature
        self.trials = trials
        self.reset = reset
        self.multi = multi
        
        self.obj = "".join(self.objectives)
        
        # 전체 실험 결과 저장용
        self.all_experiments = {}  # {problemId: {trial: {gen: stats}}}
    
    def ratio(self, numerator: float, denominator: float) -> float:
        if numerator == 0 or denominator == 0:
            return 0.0
        value = ETC.divide(numerator, denominator)
        abs_value = abs(value)
        decimal_places = max(3, -int(math.floor(math.log10(abs_value))))
        return round(value, decimal_places)
        
    def __save_results(self, trial:int, problemId:int, 
                       buggys:Programs, references:Programs, 
                       results:dict[str, dict[int, Programs]], 
                       fitness:Fitness, select:Selection) -> dict:
        
        # Save Results
        final = []
        generation_stats = {}
        
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
                    scores = fitness.run(buggy, patch)
                    scoring[patch.id] = scores
                if not scoring: continue
                sol_id = select.hype(scoring)
                patch = solutions.get_prog_by_id(sol_id)
                # Evaluation
                ## accuracy
                generation_stats[gen]['accuracy'] += 1
                
                ## similarity
                refer_sim = TED.compute_sim(buggy, refer)
                patch_sim = TED.compute_sim(buggy, patch)
                generation_stats[gen]['similarity'] += self.ratio(
                    (refer_sim - patch_sim), (refer_sim + patch_sim))
                
                ## efficiency
                refer_time = refer.results.runtime()
                patch_time = patch.results.runtime()
                generation_stats[gen]['runtime'] += self.ratio(
                    (refer_time - patch_time), (refer_time + patch_time))
                
                refer_mem = refer.results.memory()
                patch_mem = patch.results.memory()
                generation_stats[gen]['memory'] += self.ratio(
                    (refer_mem - patch_mem), (refer_mem + patch_mem))
                
                generation_stats[gen]['count'] += 1
                
                # 마지막 generation의 best solution만 final에 저장
                if gen == max(result.keys()):
                    final.append((buggy, refer, patch))
                         
        table = PrettyTable([
            "#Generation",
            "#Solutions",
            "#Fixed",
            "%Accuracy",
            "%Similarity",
            "%Runtime",
            "%Memory"
        ])
        total_bugs = len([r for r in results.values() if r])
        for gen in sorted(generation_stats.keys()):
            stats = generation_stats[gen]
            count = stats['count']
            
            if count > 0:
                table.add_row([
                    gen,
                    stats['num_solutions'],
                    count,
                    f"{(stats['accuracy'] / total_bugs * 100):.2f}%",
                    f"{(stats['similarity'] / count * 100):.2f}%",
                    f"{(stats['runtime'] / count * 100):.2f}%",
                    f"{(stats['memory'] / count * 100):.2f}%"
                ])
            else:
                table.add_row([
                    gen,
                    stats['num_solutions'],
                    0,
                    "0.00%",
                    "0.00%",
                    "0.00%",
                    "0.00%"
                ])
        
        print(table)
                
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
        
        results_path = f'results/{problemId}/{self.selection}/solutions_{trial}.csv'
        df.to_csv(results_path, index=False)
        
        # generation_stats에 total_bugs 정보 추가
        for gen in generation_stats:
            generation_stats[gen]['total_bugs'] = total_bugs
        
        return generation_stats

    def __save_experiments(self, problemId:int) -> None:
        """전체 실험(모든 trial)에 대한 통계를 계산하고 저장"""
        # Generation별 전체 trial 통계 집계
        aggregated_stats = {}
        
        for trial, gen_stats in self.all_experiments[problemId].items():
            for gen, stats in gen_stats.items():
                if gen not in aggregated_stats:
                    aggregated_stats[gen] = {
                        'accuracy': [],
                        'similarity': [],
                        'runtime': [],
                        'memory': [],
                        'num_solutions': [],
                        'count': [],
                        'total_bugs': []
                    }
                
                # 각 trial의 데이터를 리스트에 추가
                aggregated_stats[gen]['accuracy'].append(stats['accuracy'])
                aggregated_stats[gen]['similarity'].append(stats['similarity'])
                aggregated_stats[gen]['runtime'].append(stats['runtime'])
                aggregated_stats[gen]['memory'].append(stats['memory'])
                aggregated_stats[gen]['num_solutions'].append(stats['num_solutions'])
                aggregated_stats[gen]['count'].append(stats['count'])
                aggregated_stats[gen]['total_bugs'].append(stats['total_bugs'])
        
        # 평균 계산 및 테이블 생성
        summary_table = PrettyTable([
            "#Generation",
            "Avg #Solutions",
            "Avg #Fixed",
            "Avg %Accuracy",
            "Avg %Similarity",
            "Avg %Runtime",
            "Avg %Memory"
        ])
        
        # CSV 저장용 데이터
        csv_data = []
        
        for gen in sorted(aggregated_stats.keys()):
            stats = aggregated_stats[gen]
            
            avg_solutions = sum(stats['num_solutions']) / len(stats['num_solutions'])
            avg_fixed = sum(stats['count']) / len(stats['count'])
            avg_total_bugs = sum(stats['total_bugs']) / len(stats['total_bugs'])
            
            # Accuracy 계산 (각 trial의 accuracy rate를 평균)
            accuracy_rates = [
                (acc / total * 100) if total > 0 else 0 
                for acc, total in zip(stats['accuracy'], stats['total_bugs'])
            ]
            avg_accuracy = sum(accuracy_rates) / len(accuracy_rates) if accuracy_rates else 0
            
            # 다른 메트릭들의 평균 계산
            avg_similarity = sum(
                [sim / cnt * 100 if cnt > 0 else 0 
                 for sim, cnt in zip(stats['similarity'], stats['count'])]
            ) / len(stats['count']) if stats['count'] else 0
            
            avg_runtime = sum(
                [rt / cnt * 100 if cnt > 0 else 0 
                 for rt, cnt in zip(stats['runtime'], stats['count'])]
            ) / len(stats['count']) if stats['count'] else 0
            
            avg_memory = sum(
                [mem / cnt * 100 if cnt > 0 else 0 
                 for mem, cnt in zip(stats['memory'], stats['count'])]
            ) / len(stats['count']) if stats['count'] else 0
            
            summary_table.add_row([
                gen,
                f"{avg_solutions:.2f}",
                f"{avg_fixed:.2f}",
                f"{avg_accuracy:.2f}%",
                f"{avg_similarity:.2f}%",
                f"{avg_runtime:.2f}%",
                f"{avg_memory:.2f}%"
            ])
            
            # CSV 데이터 추가
            csv_data.append({
                'Generation': gen,
                'Avg_Solutions': avg_solutions,
                'Avg_Fixed': avg_fixed,
                'Avg_Accuracy': avg_accuracy,
                'Avg_Similarity': avg_similarity,
                'Avg_Runtime': avg_runtime,
                'Avg_Memory': avg_memory,
                'Trials': len(stats['count'])
            })
        
        # 결과 출력
        print(summary_table)
        
        # CSV로 저장
        summary_df = pd.DataFrame(csv_data)
        summary_path = f'results/{problemId}/{self.selection}/summary_all_trials.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Trial별 상세 결과도 저장
        detailed_data = []
        for trial in sorted(self.all_experiments[problemId].keys()):
            gen_stats = self.all_experiments[problemId][trial]
            for gen in sorted(gen_stats.keys()):
                stats = gen_stats[gen]
                total_bugs = stats['total_bugs']
                count = stats['count']
                
                detailed_data.append({
                    'Trial': trial,
                    'Generation': gen,
                    'Solutions': stats['num_solutions'],
                    'Fixed': count,
                    'Total_Bugs': total_bugs,
                    'Accuracy': (stats['accuracy'] / total_bugs * 100) if total_bugs > 0 else 0,
                    'Similarity': (stats['similarity'] / count * 100) if count > 0 else 0,
                    'Runtime': (stats['runtime'] / count * 100) if count > 0 else 0,
                    'Memory': (stats['memory'] / count * 100) if count > 0 else 0
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = f'results/{problemId}/{self.selection}/detailed_all_trials.csv'
        detailed_df.to_csv(detailed_path, index=False)

    def __core(self, trial:int, problemId:int, description:str,
               buggys:Programs, references:Programs, testcases:list):
        # Generate Feedback
        Tester.init_globals(testcases, self.timelimit)
        ga = GA(buggys, references, description,
                self.llm, self.temperature, self.objectives)
        start_time = time.process_time()
        # Run MooRepair
        results = ga.run(self.generations, self.pop_size, 
                            self.selection, self.threshold)
        time_taken = time.process_time() - start_time
        
        # 결과 저장 및 통계 반환
        generation_stats = self.__save_results(
            trial, problemId, buggys, references, 
            results, ga.fitness, ga.select
        )
        
        return generation_stats, time_taken
            
    def run(self, problems:list):
        for problem in problems:
            problemId, description, buggys, \
                references, testcases = self.setup.run(problem)
            
            # 해당 problem의 실험 데이터 초기화
            self.all_experiments[problemId] = {}
            
            for trial in range(1, self.trials+1):
                generation_stats, time_taken = self.__core(
                    trial, problemId, description,
                    buggys, references, testcases
                )
                
                # 결과 저장
                self.all_experiments[problemId][trial] = generation_stats
            
            # 전체 실험 결과 요약
            self.__save_experiments(problemId)