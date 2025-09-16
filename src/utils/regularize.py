import ast
import re

class Regularize:
    @classmethod
    def __remove_comments_and_docstrings(cls, code:str) -> str:
        # Remove single-line comments and strings that are used as comments
        code = re.sub(r'(?m)^\s*(#.*|\'[^\']*\'|"[^"]*")\s*$', '', code)
        # Remove multi-line comments and docstrings
        code = re.sub(r'(?s)(\'\'\'.*?\'\'\')|(""".*?""")', '', code)
        return code
    
    @classmethod
    def __normalize(cls, code:str) -> str:
        return code.replace('\r\n', '\n').replace('\r', '\n')
    
    @classmethod
    def __regular(cls, code:str) -> str:
        return ast.unparse(ast.parse(code))
    
    @classmethod
    def preprocessing(cls, code:str) -> str:
        pattern = r'(?<!\.)\bprint\s+(.*)'
        def repl(match):
            content = match.group(1).strip()
            return f'print({content})'
        
        pattern_comma = r'(?<!\.)\bprint\s+(.*?),\s*(#.*)?$'
        def repl_comma(match):
            content = match.group(1).strip()
            comment = match.group(2) if match.group(2) else ''
            return f'print({content}, end=\' \'){comment}'

        lines = code.split('\n')
        converted_lines = []
        for line in lines:
            if '#' in line:
                code_part, comment_part = line.split('#', 1)
                code_part = re.sub(pattern_comma, repl_comma, code_part)
                code_part = re.sub(pattern, repl, code_part)
                converted_line = f'{code_part}#' + comment_part
            else:
                line = re.sub(pattern_comma, repl_comma, line)
                line = re.sub(pattern, repl, line)
                converted_line = line
            converted_lines.append(converted_line)
        converted_code = '\n'.join(converted_lines)
        
        converted_code = re.sub(r'(?<!\.)\bprint\b\s*$', 'print()', converted_code, flags=re.MULTILINE)
        converted_code = re.sub(r'\braw_input\(', 'input(', converted_code)
        converted_code = re.sub(r'\bxrange\(', 'range(', converted_code)
        return converted_code

    @classmethod
    def run(cls, code:str) -> str:
        # code = cls.__normalize(code)
        # code = cls.preprocessing(code)
        code = cls.__regular(code)
        # code = cls.__remove_comments_and_docstrings(code)
        code = cls.__regular(code)
        return code
    
    
    