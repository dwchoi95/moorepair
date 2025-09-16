class Result:
    exec_traces = []
    vari_traces = {}
    line_vari_map = {}
    end_line = 0
    timeLimit = 1
    memLimit = 1000000
    test_code = ''
    input = ''
    expect = ''
    stdout = ''
    status = None
    exec_time = 0
    mem_usage = 0
    
    def __init__(self):
        self.exec_traces = self.exec_traces
        self.vari_traces = self.vari_traces
        self.line_vari_map = self.line_vari_map
        self.end_line = self.end_line
        self.timeLimit = self.timeLimit
        self.memLimit = self.memLimit
        self.test_code = self.test_code
        self.input = self.input
        self.expect = self.expect
        self.stdout = self.stdout
        self.status = self.status
        self.exec_time = self.exec_time
        self.mem_usage = self.mem_usage
    
    @classmethod
    def init_globals(self):
        self.exec_traces = []
        self.vari_traces = {}
        self.line_vari_map = {}
        self.end_line = 0
        self.timeLimit = 1
        self.memLimit = 1000000
        self.test_code = ''
        self.input = ''
        self.expect = ''
        self.stdout = ''
        self.status = None
        self.exec_time = 0
        self.mem_usage = 0

class Results:
    def __init__(self, code:str):
        self.tests = {}
        self.current_index = 0
        self.code = code
        self.quality_score = 0
        self.quality_messages = []
        self.maintain_score = 0
        self.maintain_messages = []
        self.set_list = []
        
    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self) -> tuple[int, Result]:
        for i, (key, value) in enumerate(self.tests.items()):
            if i == self.current_index:
                self.current_index += 1
                return key, value
        raise StopIteration
    
    def __len__(self):
        return len(self.tests)
    
    def __setitem__(self, key:int, value:Result):
        self.tests[key] = value
        self.set_list.append(key)
    
    def __getitem__(self, key:int) -> Result:
        return self.tests[key]
        
    def get_result(self) -> Result:
        return self.tests[self.set_list[0]]
    
    def get_warning(self) -> str:
        outputs = self.quality_messages + self.maintain_messages
        outputs = [f"* {msg}" for msg in outputs]
        if outputs:
            return '\n'.join(outputs)
        return ''