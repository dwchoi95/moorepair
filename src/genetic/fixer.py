import math
from .fitness import Fitness
from .crossover import Crossover
from .mutation import Mutation
from ..utils import Regularize, Randoms
    
class Fixer:
    def __init__(self, 
                 fitness:Fitness=Fitness()):
        self.fitness = fitness
        self.guidelines = fitness.guidelines
        self.crossover = Crossover()
        self.mutation = Mutation()
    
    def stochastic_universal_sampling(self, evaluates:dict) -> list:
        if not evaluates:
            raise ValueError("evaluates must not be empty")

        EPS = 1e-12
        items = []
        min_w = float('inf')

        for k, v in evaluates.items():
            base = float(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0
            w = 1.0 - base
            if math.isclose(w, 0.0, abs_tol=EPS) or math.isclose(base, 1.0, abs_tol=EPS):
                continue
            items.append((k, w))
            if w < min_w:
                min_w = w

        if not items:
            raise ValueError("No eligible items after excluding entries with v == 1.0 (or 1 - v == 0).")

        if min_w < 0.0:
            shift = -min_w
            items = [(k, w + shift) for k, w in items]

        total = sum(w for _, w in items)
        if total <= 0.0:
            return Randoms.choice([k for k, _ in items])

        pointer = Randoms.random() * total
        acc = 0.0
        for k, w in items:
            acc += w
            if acc >= pointer:
                return k
        return items[-1][0]
    
    def is_valid(self, code:str) -> bool:
        try:
            code = Regularize.run(code)
            if code.strip(): return True
        except SyntaxError: pass
        except Exception as e:
            # print(str(e))
            # print(code)
            # exit(1)
            pass
        return False
    
    def run(self, buggy:str, programs:list) -> list:
        # Crossover
        offsprings = self.crossover.run(programs)
        if len(offsprings) < len(programs):
            supplement = Randoms.sample(programs, k=len(programs)-len(offsprings))
            offsprings.extend(supplement)
        
        # Mutation
        pairs = []
        for offs in offsprings:
            if not self.is_valid(offs):
                offs = buggy
            
            evaluates = self.fitness.run(buggy, offs)
            obj = self.stochastic_universal_sampling(evaluates)
            guideline = self.guidelines.get(obj, None)
            if guideline is None:
                guideline = Randoms.choice(list(self.guidelines.values()))
                
            pairs.append((offs, guideline))
        mutants = self.mutation.run(pairs)
        
        variants = []
        for patch in mutants:
            if self.is_valid(patch):
                variants.append(patch)
        return variants