import os
import json
import math
import dataset
from tqdm import tqdm
from texttable import Texttable
from multiprocessing import Process
import warnings
warnings.filterwarnings('ignore')

from src.genetic import MooRepair, Fitness, Selector
from src.utils import DBKey, ETC, TED, Database, Sampling


class Experiments:
    def __init__(self, dataset:str, amount:int=100,
                 generations:int=4, pop_size:int=6, 
                 selection:str="rnsga3", threshold:float=1.0,
                 objectives:list=Fitness.OBJECTIVES, trials:int=1,
                 reset:bool=False, multi:bool=False):
        self.generations = generations
        self.amount = amount
        self.pop_size = pop_size
        self.selection = selection
        self.threshold = threshold
        self.objectives = objectives
        self.trials = trials
        self.reset = reset
        self.multi = multi
        
        self.obj = "".join(self.objectives)
        
        self.dataset_dir = os.path.dirname(dataset)
        
        self.dataset_db = Database(dataset)
        self.problem_tb = self.dataset_db.get_table('problem')
        self.submission_tb = self.dataset_db.get_table('submission')
        self.testcase_tb = self.dataset_db.get_table('testcase')
        
        self.results_tb = self.dataset_db.create_table('results')
        self.experiments_tb = self.dataset_db.create_table('experiments')
        
        
    def __setup(self, problem:dict):
        problemId = problem['id']
        title = problem['title']
        description = problem['description']
        timeLimit = problem['timeLimit']
        memLimit = problem['memLimit']
        programs = list(self.submission_tb.find(problemId=problemId))
        buggys = {}
        # corrects = {}
        for p in programs:
            p_id = p['id']
            code = p['code']
            status = p['status']
            if status == 'buggy':
                buggys[p_id] = code
            # elif self.uses and status == 'correct':
            #     corrects[p_id] = code
        if self.amount < 100:
            sampler = Sampling(buggys, self.amount)
            buggys = sampler.cluster()
        testcases = self.testcase_tb.find(problemId=problemId)
        testcases = [{'no': t['id'],
                           'input': t['input'],
                           'expect': t['expect'],
                           'hasStdIn': t.get('hasStdIn', False)} 
                          for t in testcases]
        
        optimals_dir = os.path.join(self.dataset_dir, str(problemId), "optimals")
        os.makedirs(optimals_dir, exist_ok=True)
        exp_dir = os.path.join(self.dataset_dir, str(problemId), "experiments")
        os.makedirs(exp_dir, exist_ok=True)

        return problemId, title, description, timeLimit, memLimit, buggys, testcases
        
    def delete_logs(self, problemId:int):
        self.results_tb.delete(
            problemId=problemId,
            fitness=self.obj,
            pop_size=self.pop_size,
            selection=self.selection)
        self.experiments_tb.delete(
            problemId=problemId,
            fitness=self.obj,
            pop_size=self.pop_size,
            selection=self.selection,
            trials=self.trials)
        
        dir = os.path.join(self.dataset_dir, str(problemId))
        if os.path.exists(dir):
            for (root, dirs, files) in os.walk(dir):
                for file in files:
                    path = os.path.join(root, file)
                    if file.startswith(f"{self.selection}_{self.pop_size}_{self.generations}"):
                        os.remove(path)
        
    
    def ratio(self, numerator: float, denominator: float) -> float:
        if numerator == 0 or denominator == 0:
            return 0.0
        value = ETC.divide(numerator, denominator)
        if not math.isfinite(value):
            return 0.0
        abs_value = abs(value)
        if abs_value == 0:
            decimal_places = 3
        else:
            # figure out how many places to show so that there's at least 3 sig-figs
            decimal_places = max(3, -int(math.floor(math.log10(abs_value))))
        return round(value, decimal_places)
        
    def __save_results(self, 
                       trial:int, 
                       problemId:int,
                       buggys:dict,
                       results:dict,
                       fitness:Fitness,
                       selector:Selector) -> dict:
        # Save Summary
        optimal_solutions = {}
        for generation, result in tqdm(results.items(), desc="Save", leave=False):
            fixed = 0
            overall_rps = 0
            overall_hv = 0
            overall_objectives = {obj:0 for obj in Fitness.OBJECTIVES}
            for b_id, buggy in buggys.items():
                solutions = []
                if b_id not in result.keys():
                    for gen in range(generation-1, 0, -1):
                        result = results[gen]
                        if b_id in result.keys():
                            solutions = results[generation-1][b_id]
                            break
                else:
                    solutions = result[b_id]
                if not solutions: continue
                scoring = {}
                hypes = {}
                buggy = buggys[b_id]
                for i, patch in enumerate(solutions):
                    evals = fitness.run(buggy, patch)
                    hype = fitness.hypervolume(evals)
                    scoring[i] = evals
                    hypes[i] = hype
                sol_id = selector.hype(scoring)
                overall_hv += hypes[sol_id]
                patch = solutions[sol_id]
                fixed += 1
                
                # Save feedbacks
                evaluates = scoring[sol_id]
                rps = TED.relative_patch_size(buggy, patch)
                overall_rps += rps
                for obj in Fitness.OBJECTIVES:
                    overall_objectives[obj] += evaluates[obj]
                
                # Save optimal solutions
                optimal_solutions[b_id] = [buggy, patch]
        
            overall_objectives = {obj: self.ratio(value, len(buggys)) 
                              for obj, value in overall_objectives.items()}
            summary = {
                DBKey.problemId: problemId,
                DBKey.fitness: self.obj,
                DBKey.generations: generation,
                DBKey.pop_size: self.pop_size,
                DBKey.selection: self.selection,
                DBKey.trials: trial,
                DBKey.b_progs: len(buggys),
                DBKey.solutions: fixed,
                DBKey.rr: self.ratio(fixed, len(buggys)),
                DBKey.rps: self.ratio(overall_rps, fixed),
                DBKey.hv: ETC.divide(overall_hv, len(buggys))
            }
            summary.update(overall_objectives)
            self.results_tb.insert(summary)
        
        # Save optimal solutions
        optimal_path = os.path.join(
            self.dataset_dir, 
            str(problemId), 
            "optimals", 
            f"{self.selection}_{self.pop_size}_{self.generations}_{trial}.json")
        with open(optimal_path, 'w') as f:
            json.dump(optimal_solutions, f, indent=2)
        
        
    def __save_experiments(self, problemId:int, title:str, buggys:dict) -> dict:
        # Performance
        experiment = self.experiments_tb.find_one(
            problemId=problemId,
            fitness=self.obj,
            pop_size=self.pop_size,
            selection=self.selection,
            trials=self.trials,
            buggy_programs=len(buggys))
        if experiment: return experiment['id']
        
        optimal_solutions_cnt = 0
        final_optimal_solutions = {}
        for generation in range(1, self.generations+1):
            tot_solution = 0
            tot_rps = 0
            tot_hv = 0
            tot_objectives = {obj:0 for obj in Fitness.OBJECTIVES}
            for result in self.results_tb.find(problemId=problemId,
                                               fitness=self.obj,
                                               generations=generation,
                                               pop_size=self.pop_size,
                                               selection=self.selection,
                                               buggy_programs=len(buggys)):
                trial = result[DBKey.trials]
                optimal_path = os.path.join(
                    self.dataset_dir, 
                    str(problemId), 
                    "optimals", 
                    f"{self.selection}_{self.pop_size}_{self.generations}_{trial}.json")
                optimal_solutions = json.load(open(optimal_path, 'r'))
                if len(optimal_solutions) > optimal_solutions_cnt:
                    optimal_solutions_cnt = len(optimal_solutions)
                    final_optimal_solutions = optimal_solutions
                
                solutions = result[DBKey.solutions]
                tot_solution += solutions
                tot_rps += result[DBKey.rps]
                tot_hv += result[DBKey.hv]
                for obj in Fitness.OBJECTIVES:
                    tot_objectives[obj] += result[obj]
        
            avg_sol = math.ceil(self.ratio(tot_solution, self.trials))
            tot_objectives = {obj: self.ratio(value, self.trials) 
                            for obj, value in tot_objectives.items()}
            
            experiments = {
                DBKey.problemId: problemId,
                DBKey.title: title,
                DBKey.fitness: self.obj,
                DBKey.generations: generation,
                DBKey.pop_size: self.pop_size,
                DBKey.selection: self.selection,
                DBKey.trials: self.trials,
                DBKey.b_progs: len(buggys),
                DBKey.a_sol: avg_sol,
                DBKey.a_rr: f'{self.ratio(avg_sol, len(buggys)):.2f}',
                DBKey.a_rps: f'{self.ratio(tot_rps, self.trials):.2f}',
                DBKey.a_hv: f'{ETC.divide(tot_hv, self.trials):.5f}',
            }
            experiments.update(tot_objectives)
            
            ## Save performance
            experiments_id = self.experiments_tb.insert(experiments)
        # Save final optimal solutions
        experiments_path = os.path.join(
            self.dataset_dir, 
            str(problemId), 
            "experiments",
            f"{self.selection}_{self.pop_size}_{self.generations}.json")
        with open(experiments_path, 'w') as f:
            json.dump(final_optimal_solutions, f, indent=2)
            
        return experiments_id
    
    def __print_table(self, table:dataset.Table, key:str):
        ## Print table
        data = table.find_one(id=key)
        tt = Texttable()
        tt.add_rows([[k,v] for k,v in data.items()])
        print(tt.draw())
        print()

    def __core(self, trial:int, problemId:int, buggys:dict, testcases:list,
               timeLimit:int, memLimit:int, title:str, description:str):
        # Generate Feedback
        result = self.results_tb.find_one(problemId=problemId,
                                          fitness=self.obj,
                                          pop_size=self.pop_size,
                                          selection=self.selection,
                                          trials=trial,
                                          buggy_programs=len(buggys))
        if result: return None
        
        # Run MooRepair
        moorepair = MooRepair(buggys=buggys.copy(), 
                              testcases=testcases,
                              timeLimit=timeLimit,
                              memLimit=memLimit,
                              title=title,
                              description=description,
                              objectives=self.objectives)
        solutions = moorepair.run(self.generations, self.pop_size, 
                                 self.selection, self.threshold)
        self.__save_results(trial, problemId, buggys, solutions, 
                            moorepair.fitness, moorepair.selector)
        # db.close()
    
    def run(self, problems:list=None):
        if problems:
            problems = [self.problem_tb.find_one(id=p) for p in problems]
        else:
            problems = list(self.problem_tb.find(order_by='id'))
        # Randoms.shuffle(problems)
        
        # if self.multi:
        #     procs = []
        #     exp_datas = []
        #     for problem in problems:
        #         problemId, title, description, timeLimit, \
        #         memLimit, buggys, testcases = self.__setup(problem)
        #         if self.reset: self.delete_logs(problemId)
        #         for trial in range(1, self.trials+1):
        #             proc = Process(target=self.__core, 
        #                         args=(trial, problemId, buggys, testcases,
        #                                 timeLimit, memLimit, title, description))
        #             proc.start()
        #             procs.append(proc)
        #         exp_datas.append((problemId, title, buggys))
        #     for proc in procs: proc.join()

        #     for problemId, title, buggys in exp_datas:
        #         experiments_id = self.__save_experiments(problemId, title, buggys)
        #         self.__print_table(self.experiments_tb, experiments_id)
        # else:
        #     for problem in problems:
        #         problemId, title, description, timeLimit, \
        #         memLimit, buggys, testcases = self.__setup(problem)
        #         if self.reset: self.delete_logs(problemId)
        #         for trial in range(1, self.trials+1):
        #             self.__core(trial, problemId, buggys, testcases,
        #                         timeLimit, memLimit, title, description)
        #         experiments_id = self.__save_experiments(problemId, title, buggys)
        #         self.__print_table(self.experiments_tb, experiments_id)
        
        for problem in problems:
            problemId, title, description, timeLimit, \
            memLimit, buggys, testcases = self.__setup(problem)
            # if self.reset: self.delete_logs(problemId)
            # for trial in range(1, self.trials+1):
            for trial in range(1, 3):
                self.__core(trial, problemId, buggys, testcases,
                            timeLimit, memLimit, title, description)
            experiments_id = self.__save_experiments(problemId, title, buggys)
            self.__print_table(self.experiments_tb, experiments_id)
    