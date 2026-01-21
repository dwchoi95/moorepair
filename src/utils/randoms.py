import random


class Randoms:
    seed = None
    
    @classmethod
    def choice(cls, elements):
        random.seed(cls.seed)
        return random.choice(elements)
    
    @classmethod
    def sample(cls, elements, k):
        random.seed(cls.seed)
        return random.sample(elements, k)

    @classmethod
    def shuffle(cls, elements):
        random.seed(cls.seed)
        return random.shuffle(elements)
    
    @classmethod
    def uniform(cls, start, end):
        random.seed(cls.seed)
        return random.uniform(start, end)
    
    @classmethod
    def random(cls):
        random.seed(cls.seed)
        return random.random()