import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from .selection import Selection
from .variation import Variation
from .fitness import Fitness
from ..execution import Programs, Program, Tester
from ..utils import ETC, TED

class GeneticAlgorithm:
    def __init__(self,
        buggys:Programs,
        references:Programs,
        description:str,
        fitness:Fitness=Fitness(),
        log_path:str="logs/temp.log",
    ):
        self.buggys = buggys
        self.references = references
        
        self.fitness = fitness
        self.select = Selection(fitness)
        self.variation = Variation(fitness, description)
        
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("GA")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.propagate = False
        
    
    def _init_population(self, buggy:Program, pop_size:int) -> Programs:
        population = Programs([ref for ref in self.references if ref.ext == buggy.ext])
        tbar = tqdm(total=pop_size, desc="Population", position=1, leave=False)
        while len(population) < pop_size:
            # Generate Patch
            left_pop = pop_size - len(population)
            programs = Programs([buggy] * left_pop)
            patches = self.variation.run(buggy, programs)
            for patch in patches:
                patch.id = f"pop_{len(population)+1}"
                population.append(patch)
                tbar.update(1)
        tbar.close()
        return population
    
    def _ga_run(self, buggy:Program, generations:int, 
                pop_size:int, selection:str, threshold:int) -> dict:
        result = {}
        early_stop = False
        solutions = Programs()
        population = self._init_population(buggy, pop_size)
        population_codes = {pop.code for pop in population}
        
        refer = self.references.get_prog_by_id(buggy.id)
        Tester.run(refer)
        ted = TED(buggy.ext)
        refer_sim = ted.compute_levenshtein_led(buggy.code, refer.code)
        refer_time = refer.results.exec_time()
        refer_mem = refer.results.mem_usage()
        
        self.logger.info(f"Buggy: {buggy.id}\n{buggy.code}\n")
        for gen in tqdm(range(1, generations+1), desc="Generation", position=1, leave=False):
            if early_stop: break
            
            result.setdefault(gen, solutions.copy())
            
            # Selection
            parents = self.select.run(buggy, population, pop_size, selection)
            
            # LLM-guided Variation
            childs = self.variation.run(buggy, parents)
                
            # Update Population
            for child in tqdm(childs, desc="Evaluation", position=2, leave=False):
                if child.code not in population_codes:
                    child.id = f"pop_{len(population)+1}"
                    population.append(child)
                    population_codes.add(child.code)
                
                # Validation
                results = Tester.run(child)
                if not Tester.is_all_pass(results): continue
                
                # Add to Solutions
                solutions.append(child)

                # Early Stop Criterion Check
                ## similarity
                patch_sim = ted.compute_levenshtein_led(buggy.code, child.code)
                sim = ETC.divide(
                    (refer_sim - patch_sim), (refer_sim + patch_sim))
                ## execution time
                patch_time = child.results.exec_time()
                exec_time = ETC.divide(
                    (refer_time - patch_time), (refer_time + patch_time))
                ## memory usage
                patch_mem = child.results.mem_usage()
                mem_usage = ETC.divide(
                    (refer_mem - patch_mem), (refer_mem + patch_mem))
                if sim >= 0 and exec_time >= 0 and mem_usage >= 0:
                    early_stop = True
                self.logger.info(f"GEN {gen} | SOL: {len(solutions)} | SIM: {sim:.2f} | ET: {exec_time:.2f} | MEM: {mem_usage:.2f}")
                self.logger.info(f"Patch:\n{child.code}\n")
                
        return result
        
    def run(self, generations:int=3, pop_size:int=10, 
            selection:str="nsga3", threshold:int=5) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._ga_run(buggy, generations, 
                pop_size, selection, threshold)
        return results
    