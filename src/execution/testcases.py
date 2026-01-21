from dataclasses import dataclass

@dataclass
class TestCase:
    id: int
    input: str
    output: str
    
    def __hash__(self):
        return hash((self.id, self.input, self.output))
    
    def __eq__(self, other):
        if not isinstance(other, TestCase):
            return False
        return self.id == other.id and self.input == other.input and self.output == other.output

class TestCases:
    def __init__(self, testcases:list):
        self.testcases = [TestCase(**tc) for tc in sorted(testcases, key=lambda x: x['id'])]
        self.current_index = 0
        
    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index < len(self.testcases):
            tc = self.testcases[self.current_index]
            self.current_index += 1
            return tc
        raise StopIteration
    
    def __len__(self):
        return len(self.testcases)
    
    def __str__(self):
        prints = ''
        for tc in self.testcases:
            prints += self.__print(tc) + '\n\n'
        return prints.strip()
    
    def __print(self, tc:TestCase) -> str:
        prints = f'[Test Case Input {tc.id}]\n{tc.input}\n'
        prints += f'[Test Case Output {tc.id}]\n{tc.output}\n'
        return prints
    
    def get_tc_id_list(self) -> list:
        return [tc.id for tc in self.testcases]
    
    def get_tc_by_id(self, id:int) -> TestCase:
        for tc in self.testcases:
            if tc.id == id:
                return tc
        raise IndexError
    