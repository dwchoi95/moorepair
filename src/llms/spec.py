import warnings
warnings.filterwarnings('ignore')

import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoTokenizer

from .ollama import Ollama
from .openai import OpenAI


@dataclass(frozen=True)
class Spec:
    name = None
    model = None
    tokenizer = None
    temperature = None
    tokenLimit = None
    
    @classmethod
    def set(cls, name:str=None, temperature:float=None):
        cls.name = name
        cls.temperature = temperature
        
        from dotenv import load_dotenv
        load_dotenv()
        MODELS_YAML_PATH = os.getenv("MODELS_YAML_PATH", "models.yaml")
        yaml_path = Path(MODELS_YAML_PATH)
        with yaml_path.open("r") as f:
            cfg = yaml.safe_load(f)
        
        if name is None:
            pconf = (cfg.get("defaults") or {})
            provider = pconf.get("provider")
            cls.name = pconf.get("name")
        else:
            providers = (cfg.get("providers") or {})
            if name.startswith("gpt"):
                provider = "openai"
                pconf = providers.get("openai", {}).get(name, {})
            else:
                provider = "ollama"
                pconf = providers.get("ollama", {}).get(name, {})
        
        if provider == "openai":
            cls.model = OpenAI(cls.name, temperature)
        else:
            cls.model = Ollama(cls.name, temperature)
            
        cls.tokenizer = AutoTokenizer.from_pretrained(pconf.get("tokenizer"))
        cls.context_window = pconf.get("context_window")
        