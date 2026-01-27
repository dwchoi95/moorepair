from .fitness import Fitness
from .crossover import Crossover
from .mutation import Mutation

from ..execution import Tester, Program, Programs
from ..utils import Randoms
    
class Variation:
    def __init__(self, 
                 fitness:Fitness,
                 description:str=""):
        self.description = description
        self.crossover = Crossover(fitness, description)
        self.mutation = Mutation(fitness, description)
    
    def cross_validation(self, refer_1:str, refer_2:str, offspring:str) -> bool:
        from ..llms import Tokenizer
        tok_1 = Tokenizer.parse(refer_1)
        tok_2 = Tokenizer.parse(refer_2)
        tok_set = set(tok_1 + tok_2)
        tok_off = Tokenizer.parse(offspring)
        for tok in set(tok_off):
            if tok not in tok_set:
                return False
        return True
    
    def tests_validation(self, program:Program) -> bool:
        Tester.run(program)
        return Tester.is_all_pass(program)
    
    def run(self, buggy:Program, programs:Programs) -> Programs:
        # Crossover
        offsprings = Programs()
        childs: Programs = self.crossover.run(buggy, programs)
        for child in childs:
            parent1= child.meta.get("parent1")
            parent2= child.meta.get("parent2")
            if not self.cross_validation(
                refer_1=programs.get_prog_by_id(parent1).code,
                refer_2=programs.get_prog_by_id(parent2).code,
                offspring=child.code
            ):
                continue
            if not self.tests_validation(child):
                continue
            offsprings.append(child)
        if len(offsprings) < len(programs):
            supplement = Randoms.sample(list(programs), k=len(programs)-len(offsprings))
            offsprings.extend(supplement)
        
        # Mutation
        mutants = Programs()
        childs: Programs = self.mutation.run(buggy, offsprings)
        for child in childs:
            if not self.tests_validation(child):
                continue
            mutants.append(child)
        return mutants