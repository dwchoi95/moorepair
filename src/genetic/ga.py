from tqdm import tqdm

from .selection import Selection
from .variation import Variation
from .fitness import Fitness
from ..llms import OpenAI, Ollama
from ..execution import Tester, Programs, Program

class GeneticAlgorithm:
    def __init__(self,
        buggys:Programs,
        references:Programs,
        description:str,
        llm:str="gpt-3.5-turbo",
        temperature:float=0.8,
        objectives:list=Fitness.OBJECTIVES
    ):
        self.buggys = buggys
        self.references = references
        self.objectives = objectives
        
        model = self._select_model(llm, temperature)
        self.fitness = Fitness(objectives)
        self.select = Selection(self.fitness)
        self.variation = Variation(model, description, self.fitness)
    
    def _select_model(self, llm:str, temperature:float):
        if llm.startswith("gpt"):
            return OpenAI(llm, temperature)
        return Ollama(llm, temperature)
    
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
        
    def _validation(self, patch:Program) -> bool:
        # Check the patch is solution
        try:
            Tester.run(patch)
            if Tester.is_all_pass(patch):
                return True
        except Exception as e:
            print(e)
            print(patch.code)
            print(patch.ext)
        return False
    
    def _ga_run(self, buggy:Program, generations:int, 
                pop_size:int, selection:str, threshold:int) -> dict:
        result = {}
        early_stop = False
        population = self._init_population(buggy, pop_size)
        for gen in tqdm(range(1, generations+1), desc="Generation", position=1, leave=False):
            solutions = Programs()
            result.setdefault(gen, solutions)
            
            # Selection
            parents = self.select.run(buggy, population, pop_size, selection)
            
            # Crossover & Mutation
            childs = self.variation.run(buggy, parents)
                
            # Update Population
            for child in childs:
                # Duplicate Check
                if any(child.code == pop.code for pop in population): continue
                child.id = f"pop_{len(population)+1}"
                population.append(child)

            # Add Solutions
            for patch in tqdm(population, desc="Evaluation", position=2, leave=False):
                if any(patch.code == sol.code for sol in solutions): continue
                
                # Patch Validation
                passed = self._validation(patch)
                if not passed: continue
                solutions.append(patch)
                
                # Early Stop Criterion Check
                if len(solutions) >= threshold:
                    early_stop = True
                if early_stop: continue
                
            if early_stop: break
        return result
        
    def run(self, generations:int=3, pop_size:int=10, 
            selection:str="nsga3", threshold:int=5) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._ga_run(buggy, generations, 
                pop_size, selection, threshold)
        return results
    