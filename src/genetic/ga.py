import logging
from pathlib import Path
from tqdm import tqdm

from .selection import Selection
from .variation import Variation
from .fitness import Fitness
from ..execution import Programs, Program, Tester
from ..utils import ETC


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
        self.variation = Variation(description)
        self._patch_uid = 0
        
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("GA")
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
            h.close()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.propagate = False

    def _assign_patch_id(self, patch: Program) -> None:
        self._patch_uid += 1
        patch.id = f"pop_{self._patch_uid}"
    
    def _is_uniq(self, new:Program, population:list[Program]) -> bool:
        new_code = ETC.normalize_code(new.code)
        for p in population:
            old_code = ETC.normalize_code(p.code)
            if old_code == new_code:
                return False
        return True
    
    def _init_population(self, buggy:Program, pop_size:int) -> list[Program]:
        population = []
        tbar = tqdm(total=pop_size, desc="Population", position=1, leave=False)
        while len(population) < pop_size:
            # Generate Patch
            left_pop = pop_size - len(population)
            programs = [None] * left_pop
            patches = self.variation.run(buggy, programs)
            for patch in patches:
                if self._is_uniq(patch, population):
                    self._assign_patch_id(patch)
                    population.append(patch)
                    tbar.update(1)
        tbar.close()
        return population
    
    def _ga_run(self, buggy:Program, generations:int, 
                pop_size:int, selection:str, threshold:float) -> dict:
        result = {}
        early_stop = False
        solutions = []
        population = self._init_population(buggy, pop_size)
        # for pop in population:
        #     results = Tester.run(pop)
        #     passed, failed = Tester.tests_split(results)
        #     self.logger.info(f"POP: {pop.id} | passed: {len(passed)}, failed: {len(failed)}\n{pop.code}\n")
        # exit()

        refer = self.references.get_prog_by_id(buggy.id)
        
        self.logger.info(f"Buggy: {buggy.id}\n{buggy.code}\n")
        for gen in tqdm(range(1, generations+1), desc="Generation", position=1, leave=False):
            if early_stop: 
                for remaining_gen in range(gen, generations+1):
                    result.setdefault(remaining_gen, solutions.copy())
                break
            result.setdefault(gen, solutions.copy())
            self.logger.info(f"=== Generation {gen} ===")
            
            # Selection
            parents = self.select.pairs(buggy, population)
            
            # LLM-guided Variation
            childs = self.variation.run(buggy, parents)
            
            # Validation
            for child in tqdm(childs, desc="Validation", position=2, leave=False):
                ## Update Population
                if self._is_uniq(child, population):
                    self._assign_patch_id(child)
                    population.append(child)
                else: continue
                
                ## Update Solutions
                results = Tester.run(child)
                if not Tester.is_all_pass(results): continue
                if self._is_uniq(child, solutions):
                    solutions.append(child)
            
            # Early Stop Criterion Check
            progs = solutions.copy()
            progs.append(refer)
            scores = self.fitness.evaluate(buggy, progs)
            for i, patch in tqdm(enumerate(solutions, start=1), desc="Evaluation", position=2, leave=False):
                refer_score = scores[refer.id]
                patch_score = scores[patch.id]
                
                ## Similarity
                ### Code Coverage Distance
                refer_ccd = refer_score['f3']
                patch_ccd = patch_score['f3']
                ccd = ETC.divide(
                    (refer_ccd - patch_ccd), (refer_ccd + patch_ccd))
                ### Tree Edit Distance
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
                
                log = f"Patch {i}: CCD: {ccd:.2f} | TED: {ted:.2f} | ET: {exec_time:.2f} | MEM: {mem_usage:.2f}"
                mean = ETC.divide(ccd + ted + exec_time + mem_usage, 4)
                if mean >= threshold:
                    early_stop = True
                    log += f" >>> Early Stopped!"
                log += f"\n{patch.code}\n"
                self.logger.info(log)

            # Replacement
            population = self.select.replacement(buggy, population, pop_size, selection)
            
        return result
        
    def run(self, generations:int=3, pop_size:int=10, 
            selection:str="nsga3", threshold:float=0.5) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._ga_run(buggy, generations, 
                pop_size, selection, threshold)
            break
        return results
    
