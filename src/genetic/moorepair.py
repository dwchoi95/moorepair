from tqdm import tqdm

from .selector import Selector
from .fixer import Fixer
from .pydex import PyDex
from .fitness import Fitness
from ..execution import Tester
from ..utils import TinyDatabase, Log


class MooRepair:
    def __init__(self,
                 buggys:dict,
                 testcases:list,
                 timeLimit:int=1,
                 memLimit:int=256,
                 title:str="",
                 description:str="",
                 objectives:list=Fitness.OBJECTIVES,
                 log_path:str=None):
        Tester.init_globals(testcases, timeLimit, memLimit, title, description)
        
        self.buggys = buggys
        self.objectives = objectives
        self.fitness = Fitness(objectives)
        self.selector = Selector(self.fitness)
        self.fixer = Fixer(self.fitness)
        self.pydex = PyDex()
        self.db = TinyDatabase(log_path)
    
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
                pop_size:int, selection:str, threshold:float, log:Log) -> dict:
        early_stop = False
        # Initialization
        population = self._init_population(buggy, pop_size)
        result = {gen:list() for gen in range(1, generations+1)}
        for gen in tqdm(range(1, generations+1), desc="Generation", position=1, leave=False):
            log.insert({'generation': gen, 'buggy':buggy, 'population': population})
            solutions = result[gen]
            
            # Evaluation & Selection
            parents = self.selector.run(buggy, population, pop_size, selection)
            log.update({'parents': parents})
            
            # Variation (Crossover & Mutation)
            childs = self.fixer.run(buggy, parents)
            log.update({'childs': childs})
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
                log.update({p_id: {'patch': patch, 'passed': passed}})
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
    
    def _pydex_run(self, buggy:str, generations:int, threshold:float, log:Log) -> dict:
        early_stop = False
        result = {gen:list() for gen in range(1, generations+1)}
        for gen in tqdm(range(1, generations+1), desc="Generation", position=1, leave=False):
            log.insert({'generation': gen, 'buggy':buggy})
            solutions = result[gen]
            patches = self.pydex.run(buggy)
            log.update({'patches': patches})
                
            # Replacement
            for p_id, patch in tqdm(enumerate(patches, start=1), 
                              desc="Validation",
                              position=2,
                              leave=False):
                
                # Duplicate Check
                if patch in solutions: continue
                
                # Patch Validation
                passed = self._validation(patch)
                log.update({p_id: {'patch': patch, 'passed': passed}})
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
        
        
    def run(self, generations:int, pop_size:int, 
            selection:str, threshold:float) -> dict:
        results = {gen:dict() for gen in range(1, generations+1)}
        for b_id, b_code in self.buggys.items():
            log = Log(self.db.table(str(b_id)))
            if selection == "pydex":
                result = self._pydex_run(b_code, generations, threshold, log)
            else:
                result = self._ga_run(b_code, generations, 
                                    pop_size, selection, threshold, log)
            for gen, solutions in result.items():
                results[gen][b_id] = solutions
        return results
    