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
        logs:bool=False,
    ):
        self.buggys = buggys
        self.references = references
        
        self.fitness = fitness
        self.select = Selection(fitness)
        self.variation = Variation(fitness, description)
        
        self.logs = logs
        self.logger = None
        
        # Setup logger
        if self.logs:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"ga_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            self.logger = logging.getLogger(f"GA_{id(self)}")
            self.logger.setLevel(logging.INFO)
            
            # File handler with immediate flush
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.info(f"GA run started - Log file: {log_file}")
    
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
                if early_stop: continue
                if len(solutions) >= threshold:
                    early_stop = True
            
            if self.logger:
                scoring = {}
                for patch in solutions:
                    scores = self.fitness.run(buggy, patch)
                    scoring[patch.id] = scores
                if not scoring: continue
                sol_id = self.select.hype(scoring)
                patch = solutions.get_prog_by_id(sol_id)
                refer = self.references.get_prog_by_id(buggy.id)
                ## similarity
                ted = TED(buggy.ext)
                refer_sim = ted.compute_levenshtein_led(buggy.code, refer.code)
                patch_sim = ted.compute_levenshtein_led(buggy.code, patch.code)
                sim = ETC.divide(
                    (refer_sim - patch_sim), (refer_sim + patch_sim))
                ## runtime
                refer_time = refer.results.exec_time()
                patch_time = patch.results.exec_time()
                eff = ETC.divide(
                    (refer_time - patch_time), (refer_time + patch_time))
                ## memory
                refer_mem = refer.results.mem_usage()
                patch_mem = patch.results.mem_usage()
                mem = ETC.divide(
                    (refer_mem - patch_mem), (refer_mem + patch_mem))
                
                self.logger.info(f"Gen {gen} | Solutions: {len(solutions)} | Sim: {sim:.4f} | Eff: {eff:.4f} | Mem: {mem:.4f}")
                
        return result
        
    def run(self, generations:int=3, pop_size:int=10, 
            selection:str="nsga3", threshold:int=5) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._ga_run(buggy, generations, 
                pop_size, selection, threshold)
        return results
    