import logging
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
        formatter = logging.Formatter('%(asctime)s - %(message)s')
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
                pop_size:int, selection:str, threshold:float) -> dict:
        result = {}
        early_stop = False
        solutions = Programs()
        population = self._init_population(buggy, pop_size)

        refer = self.references.get_prog_by_id(buggy.id)
        
        self.logger.info(f"Buggy: {buggy.id}\n{buggy.code}\n")
        for gen in tqdm(range(1, generations+1), desc="Generation", position=1, leave=False):
            if early_stop: break
            result.setdefault(gen, solutions.copy())
            self.logger.info(f"=== Generation {gen} ===")
            
            # Selection
            parents = self.select.run(buggy, population, pop_size, selection)
            
            # LLM-guided Variation
            childs = self.variation.run(buggy, parents)
            progs = childs.copy()
            progs.append(refer)
            scores = self.fitness.evaluate(buggy, progs)
                
            # Update Population
            for child in tqdm(childs, desc="Evaluation", position=2, leave=False):
                refer_score = scores[refer.id]
                patch_score = scores[child.id]
                
                child.id = f"pop_{len(population)+1}"
                population.append(child)
            
                # Early Stop Criterion Check

                ## Correctness
                ### Validation
                results = Tester.run(child)
                if not Tester.is_all_pass(results): continue
                
                ## Similarity
                ### Line-level Edit Distance
                refer_led = refer_score['f3']
                patch_led = patch_score['f3']
                led = ETC.divide(
                    (refer_led - patch_led), (refer_led + patch_led))
                ### AST-level Edit Distance
                refer_ted = refer_score['f4']
                patch_ted = patch_score['f4']
                ted = ETC.divide(
                    (refer_ted - patch_ted), (refer_ted + patch_ted))
                
                ## Efficiency
                ### Execution Time
                refer_time = refer_score['f5']
                patch_time = patch_score['f5']
                exec_time = ETC.divide(
                    (refer_time - patch_time), (refer_time + patch_time))
                ## Memory Usage
                refer_mem = refer_score['f6']
                patch_mem = patch_score['f6']
                mem_usage = ETC.divide(
                    (refer_mem - patch_mem), (refer_mem + patch_mem))

                # Add to Solutions
                solutions.append(child)
                
                log = f"Patch {len(solutions)}: LED: {led:.2f} | TED: {ted:.2f} | ET: {exec_time:.2f} | MEM: {mem_usage:.2f}"
                mean = ETC.divide(led + ted + exec_time + mem_usage, 4)
                if mean >= threshold:
                    early_stop = True
                    log += f" >>> Early Stopped!"
                log += f"\n{child.code}\n"
                self.logger.info(log)
        
        return result
        
    def run(self, generations:int=3, pop_size:int=10, 
            selection:str="nsga3", threshold:float=0.5) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._ga_run(buggy, generations, 
                pop_size, selection, threshold)
        return results
    