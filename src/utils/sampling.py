import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from .randoms import Randoms


class Sampling:
    def __init__(self, programs:dict, k:int=10):
        self.programs = programs
        self.k = int(len(programs) * k / 100)
    
    def random(self) -> dict:
        Randoms.seed = 42
        samples = Randoms.sample(list(self.programs.items()), self.k)
        Randoms.seed = None
        return dict(samples)
    
    def __ast_nodes(self, code:str) -> list:
        tree = ast.parse(code)
        return [type(node).__name__ for node in ast.walk(tree)]
        
    def cluster(self) -> dict:
        node_sequences = [self.__ast_nodes(code) for code in self.programs.values()]
        corpus = [' '.join(seq) for seq in node_sequences]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)

        k = self.k
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)

        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        
        program_items = list(self.programs.items())
        sampled = {program_items[i][0]: program_items[i][1] for i in closest_indices}
        return sampled