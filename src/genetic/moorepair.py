from tqdm import tqdm
from multiprocessing import Process, Manager

from .selector import Selector
from .fixer import Fixer
from .pydex import PyDex
from .fitness import Fitness
from ..execution import Tester

class MooRepair:
    def __init__(self,
                 buggys:dict,
                 testcases:list,
                 timeLimit:int=1,
                 memLimit:int=256,
                 title:str="",
                 description:str="",
                 objectives:list=Fitness.OBJECTIVES):
        Tester.init_globals(testcases, timeLimit, memLimit, title, description)
        
        self.buggys = buggys
        self.objectives = objectives
        self.fitness = Fitness(objectives)
        self.selector = Selector(self.fitness)
        self.fixer = Fixer(self.fitness)
        self.pydex = PyDex()
    
    def _init_population(self, b_code:str, pop_size:int) -> dict:
        population = {}
        tbar = tqdm(total=pop_size-len(population), desc="Population", position=1, leave=False)
        while len(population) < pop_size:
            # Generate Patch
            left_pop = pop_size - len(population)
            programs = [b_code] * left_pop
            patches = self.fixer.run(b_code, programs)
            for patch in patches:
                population[f'pop_{len(population)+1}'] = patch
                tbar.update(1)
        tbar.close()
        return population
        
    def _validation(self, patch):
        # Check the patch is solution
        try:
            results = Tester.test(patch)
        except:
            print(patch)
            exit()
        test_hist = Tester.get_test_hist(results)
        if Tester.is_all_pass(test_hist):
            return True
        return False
    
    def _ga_run(self, buggy:str, generations:int, 
                pop_size:int, selection:str, threshold:float) -> dict:
        early_stop = False
        # Initialization
        population = self._init_population(buggy, pop_size)
        result = {gen:list() for gen in range(1, generations+1)}
        for gen in tqdm(range(1, generations+1), desc="Generation", position=1, leave=False):
            solutions = result[gen]
            
            # Evaluation & Selection
            parents = self.selector.run(buggy, population, pop_size, selection)
            
            # Variation (Crossover & Mutation)
            childs = self.fixer.run(buggy, parents)
            for child in childs:
                # Duplicate Check
                if child in population.values(): continue
                population[f'pop_{len(population)+1}'] = child
                
            # Replacement
            for p_id, patch in tqdm(population.items(),
                                    desc="Replacement",
                                    position=2,
                                    leave=False):
                
                # Duplicate Check
                if patch in solutions: continue
                
                # Patch Validation
                passed = self._validation(patch)
                if not passed: continue
                solutions.append(patch)
                
                # Early Stop Criterion Check
                if early_stop: continue
                
                ## Fitness Evaluation
                improvements = self.fitness.run(buggy, patch)
                if all([score >= threshold for score in improvements.values()]):
                    early_stop = True
            
            if early_stop: break
                
        return result
    
    def _pydex_run(self, buggy:str, generations:int, threshold:float) -> dict:
        early_stop = False
        result = {gen:list() for gen in range(1, generations+1)}
        for gen in tqdm(range(1, generations+1), desc="Generation", position=1, leave=False):
            solutions = result[gen]
            patches = self.pydex.run(buggy)
                
            # Replacement
            for patch in tqdm(patches, 
                              desc="Validation",
                              position=2,
                              leave=False):
                
                # Duplicate Check
                if patch in solutions: continue
                
                # Patch Validation
                passed = self._validation(patch)
                if not passed: continue
                solutions.append(patch)
                
                # Early Stop Criterion Check
                if early_stop: continue
                
                ## Fitness Evaluation
                improvements = self.fitness.run(buggy, patch)
                if all([score >= threshold for score in improvements.values()]):
                    early_stop = True
            if early_stop: break
                
        return result
    
    
    def multi_run(self, b_id:str, b_code:str, generations:int, pop_size:int,
                  selection:str, threshold:float, manager:dict):
        if selection == "pydex":
            result = self._pydex_run(b_code, generations, threshold)
        else:
            result = self._ga_run(b_code, generations, 
                                pop_size, selection, threshold)
        manager[b_id] = result
        
        
    def run(self, generations:int, pop_size:int, 
            selection:str, threshold:float) -> dict:
        
        procs = []
        manager = Manager().dict()
        results = {gen:dict() for gen in range(1, generations+1)}
        for b_id, b_code in self.buggys.items():
            p = Process(target=self.multi_run, 
                        args=(b_id, b_code, generations, pop_size, selection, threshold, manager))
            p.start()
            procs.append(p)
        for p in tqdm(procs, desc="Buggy", position=0): p.join()
        for b_id, result in manager.items():
            for gen, solutions in result.items():
                results[gen][b_id] = solutions
        return results
    