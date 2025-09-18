import tempfile
from tqdm import tqdm
from multiprocessing import Process, Manager, Lock

from .selector import Selector
from .fixer import Fixer
from .pydex import PyDex
from .fitness import Fitness
from ..execution import Tester
from ..utils import TinyDatabase, Log

class MooRepair:
    def __init__(self,
                 buggys:dict,
                 objectives:list=Fitness.OBJECTIVES,
                 log_file:str=None):
        
        self.buggys = buggys
        self.objectives = objectives
        self.fitness = Fitness(objectives)
        self.selector = Selector(self.fitness)
        self.fixer = Fixer(self.fitness)
        self.pydex = PyDex()
        if log_file is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
            self.db = TinyDatabase(temp_file.name, save=False)
            temp_file.close()
        else:
            self.db = TinyDatabase(log_file, save=True)
        
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
        
    def _validation(self, p_id:str, patch:str, log:Log, lock:Lock) -> bool:
        # Check the patch is solution
        try:
            results = Tester.test(patch)
        except:
            print(patch)
            exit()
        test_hist = Tester.get_test_hist(results)
        if Tester.is_all_pass(test_hist):
            with lock: log.update({p_id: {"patch": patch, "passed": True}})
            return True
        history = {}
        for tc, res in results:
            history[tc] = {
                "status": str(res.status),
                "input": str(res.input),
                "expect": str(res.expect),
                "stdout": str(res.stdout)
            }
        with lock: log.update({p_id: {"patch": patch, "passed": False, "history": history}})
        return False
    
    def _ga_run(self, buggy:str, generations:int, pop_size:int, 
                selection:str, threshold:float, log:Log, lock:Lock) -> dict:
        early_stop = False
        # Initialization
        population = self._init_population(buggy, pop_size)
        result = {gen:list() for gen in range(1, generations+1)}
        for gen in tqdm(range(1, generations+1), desc="Generation", position=1, leave=False):
            with lock: log.insert({"generation":gen, "buggy":buggy, "population":population})
            solutions = result[gen]
            
            # Evaluation & Selection
            parents = self.selector.run(buggy, population, pop_size, selection)
            with lock: log.update({"parents": parents})
            
            # Variation (Crossover & Mutation)
            childs = self.fixer.run(buggy, parents)
            with lock: log.update({"childs": childs})
            for child in childs:
                # Duplicate Check
                if child in population.values(): continue
                population[f'pop_{len(population)+1}'] = child
                
            # Replacement
            for p_id, patch in tqdm(population.items(),
                                    total=len(population),
                                    desc="Replacement",
                                    position=2,
                                    leave=False):
                # Duplicate Check
                if patch in solutions: continue
                
                # Patch Validation
                passed = self._validation(p_id, patch, log, lock)
                if not passed: continue
                solutions.append(patch)
                
                # Early Stop Criterion Check
                if early_stop: continue
                
                ## Fitness Evaluation
                improvements = self.fitness.run(buggy, patch)
                if all([score >= threshold for score in improvements.values()]):
                    early_stop = True
            with lock: log.update({"solutions": solutions})
            if early_stop: break
                
        return result
    
    def _pydex_run(self, buggy:str, generations:int, threshold:float, log:Log, lock:Lock) -> dict:
        early_stop = False
        result = {gen:list() for gen in range(1, generations+1)}
        for gen in tqdm(range(1, generations+1), desc="Generation", position=1, leave=False):
            with lock: log.insert({"generation":gen, "buggy":buggy})
            solutions = result[gen]
            patches = self.pydex.run(buggy)
            with lock: log.update({"patches": patches})
                
            # Replacement
            for p_id, patch in tqdm(enumerate(patches, start=1),
                                    total=len(patches),
                                    desc="Validation",
                                    position=2,
                                    leave=False):
                # Duplicate Check
                if patch in solutions: continue
                
                # Patch Validation
                passed = self._validation(p_id, patch, log, lock)
                if not passed: continue
                solutions.append(patch)
                
                # Early Stop Criterion Check
                if early_stop: continue
                
                ## Fitness Evaluation
                improvements = self.fitness.run(buggy, patch)
                if all([score >= threshold for score in improvements.values()]):
                    early_stop = True
            with lock: log.update({"solutions": solutions})
            if early_stop: break
                
        return result
    
    def worker(self, b_id, b_code, generations, pop_size, 
               selection, threshold, return_dict, lock):
        log = Log(self.db.table(str(b_id)))
        if selection == "pydex":
            result = self._pydex_run(b_code, generations, threshold, log, lock)
        else:
            result = self._ga_run(b_code, generations, 
                                 pop_size, selection, threshold, log, lock)
        return_dict[b_id] = result
        
        
    def run(self, generations:int, pop_size:int, 
            selection:str, threshold:float) -> dict:
        lock = Lock()
        procs = []
        managaer = Manager().dict()
        results = {gen:dict() for gen in range(1, generations+1)}
        for b_id, b_code in self.buggys.items():
            p = Process(target=self.worker, args=(b_id, b_code, generations, pop_size, 
                                                  selection, threshold, managaer, lock))
            procs.append(p)
            p.start()
        for p in tqdm(procs, desc="Buggys", position=0, leave=True): p.join()
        for b_id, result in managaer.items():
            for gen, solutions in result.items():
                results[gen][b_id] = solutions
        return results
    