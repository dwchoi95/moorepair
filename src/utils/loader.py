import json

from .sampling import Sampling
from ..execution import Programs, Program, TestCases

class Loader:
    def __init__(self, sampling:bool=False, initialization:bool=False):
        self.sampling = sampling
        self.initialization = initialization
    
    def run(self, problem:str) -> tuple[str, str, Programs, Programs, TestCases]:
        dataset = json.loads(open(problem, 'r').read())
        assignment = dataset['assignment']
        problemId = assignment['id']
        description = assignment['description'].replace('\n', '  \n')
        submissions = dataset['submissions']
        references, buggys = Programs(), Programs()
        
        for sub in submissions:
            if sub["status"] == "buggy": 
                buggys.append(Program(
                    id=sub["id"], code=sub["code"], ext=sub["ext"]))
            else: references.append(Program(
                    id=sub["id"], code=sub["code"], ext=sub["ext"]))
        
        if self.sampling:
            sampler = Sampling(list(buggys))
            buggys = Programs(sampler.random())
            # sampler = Sampling(references)
            # references = sampler.random()
            references = Programs([references.get_prog_by_id(buggy.id) for buggy in buggys])
        
        if self.initialization:
            references = Programs()
        
        testcases = TestCases(dataset['test_cases'])
    
        return problemId, description, buggys, references, testcases
    