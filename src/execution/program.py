from dataclasses import dataclass, field

from .results import Results
    
@dataclass
class Program:
    id:str = field(metadata={"desc":"Unique identifier for the program"})
    code:str = field(metadata={"desc":"Source code of the program"})
    ext:str = field(metadata={"desc":"File extension indicating the programming language"})
    method:str = field(default=None, metadata={"desc":"Method Name (for Java/C# programs)"})
    results:Results = field(default=None, metadata={"desc":"Run Results after execution"})
    
    def __hash__(self):
        return hash((self.code, self.ext))
    
    def __eq__(self, other):
        if not isinstance(other, Program):
            return False
        return self.code == other.code and self.ext == other.ext
    
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
    
    def __print(self, prog:Program) -> str:
        return f'```{prog.ext}\n{prog.code}\n```'

    def get_prog_id_list(self) -> list:
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