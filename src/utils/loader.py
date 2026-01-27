import json

from .sampling import Sampling
from ..execution import Programs, Program, TestCases

class Loader:
    def __init__(self, sampling:bool=False, initialization:bool=False):
        self.sampling = sampling
        self.initialization = initialization
    
    def run(self, problem:str) -> tuple:
        dataset = json.loads(open(problem, 'r').read())
        assignment = dataset['assignment']
        problemId = assignment['id']
        description = assignment['description'].replace('\n', '  \n')
        submissions = dataset['submissions']
        references, buggys = [], []
        
        for sub in submissions:
            if sub["status"] == "buggy": 
                buggys.append(Program(
                    id=sub["id"], code=sub["code"], ext=sub["ext"]))
            else: references.append(
                Program(
                    id=sub["id"], code=sub["code"], ext=sub["ext"]))
        
        if self.sampling:
            sampler = Sampling(buggys)
            buggys = sampler.random()
            sampler = Sampling(references)
            references = sampler.random()
        
        if self.initialization:
            references = []
        
        testcases = dataset['test_cases']
    
        return problemId, description, Programs(buggys), Programs(references), TestCases(testcases)
    