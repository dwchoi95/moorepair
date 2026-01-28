from .spec import Spec


class Tokenizer:
    @staticmethod
    def parse(text:str) -> list[int]:
        tokenizer = Spec.tokenizer
        tokens = tokenizer.tokenize(text)
        return tokens
    
    @staticmethod
    def encode(text:str) -> list[int]:
        tokenizer = Spec.tokenizer
        input_ids = tokenizer.encode(text)
        return input_ids
    
    @staticmethod
    def length(text:str) -> int:
        tokenizer = Spec.tokenizer
        input_ids = tokenizer.encode(text)
        return len(input_ids)