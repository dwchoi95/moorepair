from transformers import AutoTokenizer

class Tokenizer:
    tokenizer = None

    @classmethod
    def set(cls, model_name:str):
        # LiteLLM model names (e.g., "ollama/codellama") are not HF repo
        # ids; skip silently when no matching tokenizer exists
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            cls.tokenizer = None
        
    @classmethod
    def parse(cls, text:str) -> list[int]:
        tokens = cls.tokenizer.tokenize(text)
        return tokens
    
    @classmethod
    def encode(cls, text:str) -> list[int]:
        input_ids = cls.tokenizer.encode(text)
        return input_ids
    
    @classmethod
    def length(cls, text:str) -> int:
        return len(cls.encode(text))