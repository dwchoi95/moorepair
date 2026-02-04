from dataclasses import dataclass, field

from .results import Results
from ..utils.ted import TED
    
@dataclass
class Program:
    id:str = field(metadata={"desc":"Unique identifier for the program"})
    code:str = field(metadata={"desc":"Source code of the program"})
    ext:str = field(metadata={"desc":"File extension indicating the programming language"})
    method:str = field(default=None, metadata={"desc":"Method Name (for Java/C# programs)"})
    results:Results = field(default=None, metadata={"desc":"Run Results after execution"})
    meta:dict = field(default_factory=dict, metadata={"desc":"Additional metadata"})
    
    def __hash__(self):
        seq = ["".join(line.split()) for line in self.code.splitlines() if line.strip()]
        seq2str = ''.join(seq)
        return hash((seq2str, self.ext))
    
    def __eq__(self, other):
        if not isinstance(other, Program) or self.ext != other.ext:
            return False
        ted = TED(self.ext)
        sim = ted.compute_levenshtein_led(self.code, other.code)
        return sim == 0
    
    
class Programs:
    def __init__(self, programs:list[Program]=[]):
        self.programs = [p for p in sorted(programs, key=lambda x: x.id)]
        self.current_index = 0
        
    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index < len(self.programs):
            prog = self.programs[self.current_index]
            self.current_index += 1
            return prog
        raise StopIteration
    
    def __len__(self):
        return len(self.programs)
    
    def __str__(self):
        prints = ''
        for prog in self.programs:
            prints += self.__print(prog) + '\n\n'
        return prints.strip()
    
    def __getitem__(self, idx) -> Program | list[Program]:
        if isinstance(idx, slice):
            return Programs(self.programs[idx])
        return self.programs[idx]
    
    def __print(self, prog:Program) -> str:
        return f'```{prog.ext}\n{prog.code}\n```'

    def get_prog_id_list(self) -> list[str]:
        return [prog.id for prog in self.programs]
    
    def get_prog_by_id(self, id:int) -> Program:
        for prog in self.programs:
            if prog.id == id:
                return prog
        raise IndexError
    
    def extend(self, programs:list[Program]):
        self.programs.extend(programs)
    
    def append(self, program:Program):
        self.programs.append(program)
    
    def copy(self) -> 'Programs':
        return Programs(self.programs.copy())