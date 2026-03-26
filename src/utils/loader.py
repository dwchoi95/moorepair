import json

from .sampling import Sampling
from ..execution import Programs, Program, TestCases

class Loader:
    def __init__(self, sampling:bool=False, initialization:bool=False):
        self.sampling = sampling
        self.initialization = initialization
    
    def run(self, problem:str) -> tuple[str, str, int, int, Programs, Programs, TestCases]:
        dataset = json.loads(open(problem, 'r').read())
        assignment = dataset['assignment']
        timelimit = int(assignment['time_limit'])
        memlimit = int(assignment['memory_limit'])

        submissions = dataset['submissions']
        mismatches = dataset.get('mismatches', [])
        references, buggys = Programs(), Programs()
        
        for sub in submissions:
            if sub["id"] in mismatches: continue
            if sub["status"] == "OK": 
                references.append(Program(
                    id=sub["id"], code=sub["code"], ext=sub["ext"]))
            else:
                buggys.append(Program(
                    id=sub["id"], code=sub["code"], ext=sub["ext"]))
        
        if self.sampling:
            sampler = Sampling(list(buggys))
            buggys = Programs(sampler.random())
            sampler = Sampling(list(references))
            references = Programs(sampler.random())
        
        if self.initialization:
            references = Programs()
        
        testcases = TestCases(dataset['test_cases'])


        return assignment, timelimit, memlimit, buggys, references, testcases
    