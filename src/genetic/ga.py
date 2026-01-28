from tqdm import tqdm

from .selection import Selection
from .variation import Variation
from .fitness import Fitness
from ..execution import Programs, Program, Tester

class GeneticAlgorithm:
    def __init__(self,
        buggys:Programs,
        references:Programs,
        description:str,
        fitness:Fitness=Fitness(),
    ):
        self.buggys = buggys
        self.references = references
        
        self.select = Selection(fitness)
        self.variation = Variation(fitness, description)
    
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
                
        return result
        
    def run(self, generations:int=3, pop_size:int=10, 
            selection:str="nsga3", threshold:int=5) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._ga_run(buggy, generations, 
                pop_size, selection, threshold)
        return results
    