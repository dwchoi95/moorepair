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
        description = assignment['description']
        input_format = assignment['input_format']
        output_format = assignment['output_format']
        interaction_format = assignment['interaction_format']
        note = assignment['note']
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
            # sampler = Sampling(references)
            # references = sampler.random()
            references = Programs([references.get_prog_by_id(buggy.id) for buggy in buggys])
        
        if self.initialization:
            references = Programs()
        
        testcases = TestCases(dataset['test_cases'])

        description = f"Problem Description:\n{description}"
        if input_format:
            description += f"\n\nInput Format:\n{input_format}"
        if output_format:
            description += f"\n\nOutput Format:\n{output_format}"
        if interaction_format:
            description += f"\n\nInteraction Format:\n{interaction_format}"
        if note:
            description += f"\n\nNote:\n{note}"

        return problemId, description, timelimit, memlimit, buggys, references, testcases
    