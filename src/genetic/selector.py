import numpy as np
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.moead import MOEAD
from sklearn.cluster import KMeans

from .fitness import Fitness
from ..utils import Randoms


SELECTIONS = ["none", "random", "nsga3", "moead", "rnsga3", "hype", "pydex"]

class MOO(Problem):
    def __init__(self, scores: dict):
        self.keys = list(scores.keys())
        self.n_obj = len(scores[self.keys[0]])
        self.fitness = np.array([scores[k] for k in self.keys], dtype=float)
        # set number of variables equal to number of objectives
        n_var = self.n_obj
        xl = np.array([0.0] * self.n_obj)
        xu = np.array([1.0] * self.n_obj)

        super().__init__(
            n_var=n_var,
            n_obj=self.n_obj,
            n_constr=0,
            xl=xl,
            xu=xu,
            elementwise_evaluation=False
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        dists = np.linalg.norm(self.fitness[None, :, :] - X[:, None, :], axis=2)
        idx = np.argmin(dists, axis=1)
        F = self.fitness[idx]
        out['F'] = -F

class Selector:
    def __init__(self, fitness:Fitness=Fitness()):
        self.fitness = fitness
    
    def random(self, scores:dict, pop_size:int) -> list:
        return Randoms.sample(list(scores.keys()), pop_size)
    
    def nsga3(self, scores:dict, pop_size:int, factory:str="das-dennis") -> list:
        return self.__moo_run(scores, pop_size, factory=factory, algo="nsga3")
    
    def rnsga3(self, scores:dict, pop_size:int, factory:str="custom") -> list:
        return self.__moo_run(scores, pop_size, factory=factory, algo="rnsga3")
    
    def moead(self, scores:dict, pop_size:int, factory:str="das-dennis") -> list:
        return self.__moo_run(scores, pop_size, factory=factory, algo="moead")
    
    def __moo_run(self, scores:dict, pop_size:int, factory:str="das-dennis", algo:str="rnsga3") -> list:
        # Set Problem
        problem = MOO(scores)
        
        ## Set Reference directions
        M = problem.n_obj                   # 목표 함수 개수
        N = pop_size                        # 해의 개수
        H = int(np.ceil(N ** (1 / (M-1))))  # 파티션 수
        # H = max(2, int(np.ceil(N ** (1 / max(1, M-1))))) # 최소값 보장
        if factory == "das-dennis":
            ref_dirs = get_reference_directions(factory, M, n_partitions=H)
        elif factory == "energy":
            ref_dirs = get_reference_directions(factory, M, n_points=pop_size)
        else: # custom
            objectives_array = np.array(list(scores.values()))
            n_clusters = pop_size
            final_n_clusters = n_clusters
            wcss = []
            k_range = range(1, n_clusters + 1)
            
            for k in k_range:
                if k == 1:
                    center = np.mean(objectives_array, axis=0)
                    wcss_k = np.sum((objectives_array - center) ** 2)
                else:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(objectives_array)
                    wcss_k = kmeans.inertia_
                wcss.append(wcss_k)
            
            if len(wcss) < 3:
                return 1
            
            slopes = []
            for i in range(1, len(wcss)):
                slope = wcss[i-1] - wcss[i]
                slopes.append(slope)
            
            slope_changes = []
            for i in range(1, len(slopes)):
                change = slopes[i-1] - slopes[i]
                slope_changes.append(change)
            
            if not slope_changes:
                return 2
            
            elbow_idx = np.argmax(slope_changes)
            optimal_k = elbow_idx + 2
            
            if optimal_k < n_clusters:
                final_n_clusters = optimal_k
                
            # K-means 클러스터링을 사용하여 대표점들 찾기
            if final_n_clusters == 1:
                ref_points = np.mean(objectives_array, axis=0, keepdims=True)
            else:
                kmeans = KMeans(n_clusters=final_n_clusters, random_state=42, n_init=10)
                kmeans.fit(objectives_array)
                ref_points = kmeans.cluster_centers_
    
        if algo == "moead":
            algo = MOEAD(ref_dirs, 
                         n_offsprings=pop_size,
                        #  prob_neighbor_mating=0.9,
                        #  n_neighbors=max(5, N//5)
                         )
        elif algo == "nsga3":
            algo = NSGA3(ref_dirs, 
                         n_offsprings=pop_size,
                        #  eliminate_duplicates=True
                         )
        elif algo == "rnsga3":
            algo = RNSGA3(ref_points,
                          pop_per_ref_point=pop_size)
        else:
            raise ValueError(f"Invalid algorithm: {algo}. Choose 'moead' or 'nsga3'.")
        
        res = minimize(problem, algo, verbose=False)
        X_pop = res.pop.get("X")
        
        selected = []
        used_indices = set()
        for x in X_pop:
            dists = np.linalg.norm(problem.fitness - x, axis=1)
            available_indices = [i for i in range(len(dists)) if i not in used_indices]
            if not available_indices:
                available_indices = list(range(len(dists)))
            min_idx = min(available_indices, key=lambda i: dists[i])
            used_indices.add(min_idx)
            key = problem.keys[min_idx]
            selected.append(key)
        return selected[:pop_size]
    
    def hype(self, scores:dict, pop_size:int=1) -> str:
        hv_values = {key: self.fitness.hypervolume(value)
                     for key, value in scores.items()}
        sorted_keys = sorted(hv_values, key=hv_values.get, reverse=True)
        return sorted_keys[:pop_size] if pop_size > 1 else sorted_keys[0]
    
    
    def run(self, buggy:str, references:dict, pop_size:int, selection:str="rnsga3"):
        # Fitness Evaluation
        results = {}
        for obj in self.fitness.objectives:
            results[obj] = self.fitness.OBJ_FUNC_MAP[obj](buggy, references)
        scores = {r_id: [results[obj][r_id] for obj in self.fitness.objectives]
                    for r_id in references.keys()}
        
        # Selection
        if selection == "none":
            return [buggy] * pop_size
        elif selection == "random":
            selected = self.random(scores, pop_size)
        elif selection == "nsga3":
            selected = self.nsga3(scores, pop_size)
        elif selection == "rnsga3":
            selected = self.rnsga3(scores, pop_size)
        elif selection == "moead":
            selected = self.moead(scores, pop_size)
        elif selection == "hype":
            selected = self.hype(scores, pop_size)
        elif selection == "pydex":
            raise NotImplementedError("pydex selection is not implemented in Selector.")
        else:
            raise ValueError(f"Invalid selection method: {selection}. Choose from {SELECTIONS}.")
            
        programs = [references[p_id] for p_id in selected]
        return programs
    