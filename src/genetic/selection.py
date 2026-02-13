import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.population import Population
from pymoo.indicators.hv import HV

from .fitness import Fitness
from ..utils import Randoms
from ..execution import Program


SELECTIONS = ["none", "random", "nsga2", "nsga3", "hype", "hier"]

class Selection:
    def __init__(self, fitness:Fitness):
        self.fitness = fitness
    
    def random(self, scores:dict, pop_size:int) -> list:
        return Randoms.sample(list(scores.keys()), pop_size)
    
    def nsga2(self, scores:dict, pop_size:int) -> list:
        keys = list(scores.keys())
        m = len(scores[keys[0]])
        F = np.array([scores[k] for k in keys], dtype=float)
        # F = -F # if maximize use -F to convert to minimize

        X = np.zeros((len(keys), 1))
        pop = Population.new("X", X, "F", F)
        pop.set("key", np.array(keys, dtype=object))
        problem = Problem(n_var=1, n_obj=m, xl=np.array([0.0]), xu=np.array([1.0]))

        algo = NSGA2(pop_size=pop_size)
        n_survive = min(pop_size, len(pop))
        survivors = algo.survival.do(problem, pop, n_survive=n_survive)

        return survivors.get("key").tolist()
    
    def nsga3(self, scores:dict, pop_size:int) -> list:
        keys = list(scores.keys())
        m = len(scores[keys[0]])
        F = np.array([scores[k] for k in keys], dtype=float)
        # F = -F # if maximize use -F to convert to minimize

        X = np.zeros((len(keys), 1))
        pop = Population.new("X", X, "F", F)
        pop.set("key", np.array(keys, dtype=object))
        problem = Problem(n_var=1, n_obj=m, xl=np.array([0.0]), xu=np.array([1.0]))

        ref_dirs = get_reference_directions("energy", m, m)
        algo = NSGA3(ref_dirs=ref_dirs)
        n_survive = min(pop_size, len(pop))
        survivors = algo.survival.do(problem, pop, n_survive=n_survive)

        return survivors.get("key").tolist()
    
    def hype(self, scores:dict, pop_size:int=1) -> list | str:
        hv_values = {}
        for key, value in scores.items():
            x = np.array(value, dtype=float)
            ref = np.ones(len(value), dtype=float)
            hv = HV(ref_point=ref)
            hv_values[key] = float(hv(x.reshape(1, -1)))
        sorted_keys = sorted(hv_values, key=hv_values.get, reverse=True)
        return sorted_keys[:pop_size] if pop_size > 1 else sorted_keys[0]
    
    def hierarchical(self, scores:dict, pop_size:int) -> list:
        sorted_programs = sorted(
            scores.keys(),
            key=lambda prog: (
                scores[prog][0] + scores[prog][1], # f1 + f2
                scores[prog][2] + scores[prog][3], # f3 + f4
                scores[prog][4] + scores[prog][5]  # f5 + f6
            )
        )
        return sorted_programs[:pop_size]
    
    def replacement(self, buggy:Program, references:list[Program], pop_size:int, selection:str="nsga3") -> list[Program]:
        # Fitness Evaluation
        scores = self.fitness.run(buggy, references)
        
        # Selection
        if selection == "none":
            return [buggy] * pop_size
        elif selection == "random":
            selected = self.random(scores, pop_size)
        elif selection == "nsga2":
            selected = self.nsga2(scores, pop_size)
        elif selection == "nsga3":
            selected = self.nsga3(scores, pop_size)
        elif selection == "hype":
            selected = self.hype(scores, pop_size)
        elif selection == "hier":
            selected = self.hierarchical(scores, pop_size)
        else:
            raise ValueError(f"Invalid selection method: {selection}. Choose from {SELECTIONS}.")
        
        programs = [refer for refer in references if refer.id in selected]
        return programs
    
    def __get_prog_by_id(self, p_id:str, programs:list[Program]) -> Program:
        for p in programs:
            if p.id == p_id:
                return p
        raise IndexError
        
    def pairs(self, buggy:Program, population:list[Program]) -> list:
        if population[0] is None:
            return population
        normalized = self.fitness.evaluate(buggy, population)
        obj_sorted_asc = {
            obj: sorted(normalized.keys(),
                        key=lambda ind: normalized[ind][obj])
            for obj in self.fitness.objectives
        }
        parents = []
        for ind_id, scores in normalized.items():
            parent1 = self.__get_prog_by_id(ind_id, population)
            max_obj = max(scores, key=scores.get)
            min_obj = min(scores, key=scores.get)
            strengths1 = self.fitness.strengths[min_obj]
            strengths2 = self.fitness.strengths[max_obj]
            for cand_id in obj_sorted_asc[max_obj]:
                if cand_id != ind_id:
                    parent2 = self.__get_prog_by_id(cand_id, population)
                    parents.append((parent1, parent2, strengths1, strengths2))
                    break
        return parents