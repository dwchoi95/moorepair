import dataset
from tinydb import TinyDB
from tinydb.storages import JSONStorage, MemoryStorage
from tinydb.middlewares import CachingMiddleware

class TinyDatabase(TinyDB):
    def __init__(self, *args, **kwargs):
        if kwargs.pop('save', True):
            kwargs['indent'] = 2
            kwargs['separators'] = (',', ':')
            kwargs.setdefault('storage', JSONStorage)
        else:
            args = ()
            kwargs.setdefault('storage', CachingMiddleware(MemoryStorage))
        super().__init__(*args, **kwargs)

    def get_data_from_table(self, tablename):
        return {k:v 
                for d in self.table(tablename).all() 
                for k, v in d.items()}
        
        
class Database:
    def __init__(self, db_file:str):
        self.db = dataset.connect(f"sqlite:///{db_file}")
    
    def __exit__(self):
        self.db.close()
        
    def get_table(self, name:str) -> dataset.Table:
        return self.db[name]
    
    def create_table(self, name:str) -> dataset.Table:
        if name not in self.db.tables:
            return self.db.create_table(name)
        else:
            return self.db[name]


class DBKey:
    problemId = 'problemId'
    description = 'description'
    
    problem = 'problem'
    title = 'title'
    fitness = 'fitness'
    generations = 'generations'
    pop_size = 'pop_size'
    selection = 'selection'
    
    b_progs = 'buggy_programs'
    
    trials = 'trials'
    a_sol = 'avg_solutions'
    a_rr = 'avg_rr'
    a_rps = 'avg_rps'
    a_hv = 'avg_hv'
    
    solutions = 'solutions'
    rr = 'repair_rate'
    rps = 'relative_patch_size'
    hv = 'hypervolume'
    
    f1 = 'f1'
    f2 = 'f2'
    f3 = 'f3'
    f4 = 'f4'
    f5 = 'f5'
    f6 = 'f6'
