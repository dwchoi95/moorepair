import sys
from io import StringIO
import unittest
from unittest.mock import patch, mock_open

from .timeout import timeout
from .results import Result

class UnitTestStatus:
    success = 'Success'
    error = 'Error'
    failure = 'Failure'
    
class TextTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super_class = super(TextTestResult, self)
        super_class.__init__(stream, descriptions, verbosity)
        # self.stdout = ""
    def addSuccess(self, test):
        super(TextTestResult, self).addSuccess(test)
        self.stdout = test.stdout
        self.status = UnitTestStatus.success
    def addError(self, test, err):
        super(TextTestResult, self).addError(test, err)
        self.stdout = err[1]
        self.status = UnitTestStatus.error
    def addFailure(self, test, err):
        super(TextTestResult, self).addFailure(test, err)
        self.stdout = test.stdout
        self.status = UnitTestStatus.failure

class Validating(unittest.TestCase):
    def setUp(self):
        self.original_globals = dict(globals()).copy()
        globals()['__name__'] = '__main__'
        self.input_data = StringIO(Result.input)
        self.mock_stdout = StringIO()
        self.mock_stderr = StringIO()
        self.stdout = None

    def tearDown(self):
        globals().clear()
        globals().update(self.original_globals)
        self.input_data.close()
        self.mock_stdout.close()
        self.mock_stderr.close()

    def assertEqual(self, actual, expected, msg=None):
        try:
            try:
                actual_lines = actual.splitlines()
                actual = '\n'.join(line.rstrip() for line in actual_lines)
            except: pass
            try:
                expected_lines = expected.splitlines()
                expected = '\n'.join(line.rstrip() for line in expected_lines)
            except: pass
            try: actual = eval(actual)
            except: pass
            try: expected = eval(expected)
            except: pass
            super().assertEqual(actual, expected, msg)
        except AssertionError as e:
            raise AssertionError(actual)
    
    @timeout(Result.timeLimit, 
             use_signals=False,
             timeout_exception=RuntimeError, 
             exception_message="TimeoutError")
    def exec_with_timeout(self, code):
        exec(code, globals())
            
    def test(self):
        with (
            patch('sys.stdin', self.input_data),
            patch('builtins.open', mock_open(read_data=Result.input)),
            patch('builtins.input', side_effect=lambda *a, **k: sys.stdin.readline().rstrip('\\n')),
            patch('sys.stdout', self.mock_stdout),
            patch('sys.stderr', self.mock_stderr)):
            self.exec_with_timeout(Result.test_code)
        
        self.stdout = self.mock_stdout.getvalue().strip()
        self.assertEqual(self.stdout, Result.expect)
        
    
class Tracing(Validating):
    @timeout(Result.timeLimit, 
             use_signals=False,
             timeout_exception=RuntimeError, 
             exception_message="TimeoutError")
    def exec_with_timeout(self, code, trace:bool=True):
        exec(code, globals())
            
    def test(self):
        with (
            patch('sys.stdin', self.input_data),
            patch('builtins.open', mock_open(read_data=Result.input)),
            patch('builtins.input', side_effect=lambda *a, **k: sys.stdin.readline().rstrip('\\n')),
            patch('sys.stdout', self.mock_stdout),
            patch('sys.stderr', self.mock_stderr)):
            self.exec_with_timeout(Result.test_code, trace=True)
        
        self.stdout = self.mock_stdout.getvalue().strip()
        self.assertEqual(self.stdout, Result.expect)
    
class RunUnitTest:
    @staticmethod
    def run(UnitTest:object=Validating):
        suite = unittest.TestLoader().loadTestsFromTestCase(UnitTest)
        runner = unittest.TextTestRunner(StringIO())
        runner.resultclass = TextTestResult
        res = runner.run(suite)
        Result.status = res.status
        Result.stdout = res.stdout