from .randoms import Randoms


class Sampling:
    def __init__(self, programs:list):
        self.programs = programs
        self.k = 1
        # self.k = len(programs) / 10
        # if self.k < 10: self.k = 10
    
    def random(self) -> list:
        Randoms.seed = 42
        samples = Randoms.sample(self.programs, int(self.k))
        Randoms.seed = None
        return samples
    