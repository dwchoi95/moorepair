import json

from .sampling import Sampling
from ..execution import Programs, Program

class Setup:
    def __init__(self, sampling:bool=False, initialization:bool=False):
        self.sampling = sampling
        self.initialization = initialization
    
    def run(self, problem:str) -> tuple:
        dataset = json.loads(open(problem, 'r').read())
        assignment = dataset['assignment']
        problemId = assignment['id']
        description = assignment['description']
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
            buggys = Programs(buggys)
            sampler = Sampling(references)
            references = sampler.random()
            references = Programs(references)
        if self.initialization:
            references = Programs()
                
        testcases = dataset['test_cases']
        testcases = [{
            'id': t['id'],
            'input': t['input'],
            'output': t['output'],
            } for t in testcases]
        return problemId, description, buggys, references, testcases
    