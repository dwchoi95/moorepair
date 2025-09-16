class Testcase:
    def __init__(self, testcase:dict):
        self.no = testcase.get('no', 1)
        self.input = testcase.get('input', None)
        self.expect = testcase.get('expect', None)
        self.hasStdIn = testcase.get('hasStdIn', False)

class TestSuite:
    def __init__(self, testcases:list):
        self.testcases = [Testcase(tc) for tc in sorted(testcases, key=lambda x: x['no'])]
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
        prints = "|  #  | Input | Output |\n| :-: | :---: | :------: |\n"
        for i, tc in enumerate(self.testcases, start=1):
            test_input = tc.input.replace("\n", "<br />")
            test_expect = tc.expect.replace("\n", "<br />")
            prints += f"|  {i}  | {test_input} | {test_expect} |\n"
        return prints.strip()
    
    def print_testcase(self, no:int) -> str:
        prints = ''
        for tc in self.testcases:
            if tc.no == no:
                prints += f"TestcaseNo: {no}\n"
                prints += f"Input:  \n{tc.input}  \n"
                prints += f"Expect:  \n{tc.expect}"
        return prints

    def get_tc_no_list(self) -> list:
        return [tc.no for tc in self.testcases]
    
    def get_tc_by_no(self, no:int) -> Testcase:
        for tc in self.testcases:
            if tc.no == no:
                return tc
        raise IndexError
    