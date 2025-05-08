
from functools import lru_cache
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline
)
import torch

class ModelManager:
    def __init__(self):
        self.model_registry = {
            "Bio_ClinicalBERT": {
                "path": "emilyalsentzer/Bio_ClinicalBERT",
                "type": "classification"
            },
            "Longformer": {
                "path": "allenai/longformer-base-4096",
                "type": "classification"
            },
            "BART": {
                "path": "facebook/bart-large-cnn",
                "type": "summarization"
            },
            "T5": {
                "path": "t5-base",
                "type": "summarization"
            },
            "BioGPT": {
                "path": "microsoft/biogpt",
                "type": "generation"
            },
            "GPTNeo": {
                "path": "EleutherAI/gpt-neo-125M",
                "type": "generation"
            }
        }
        self.device = 0 if torch.cuda.is_available() else -1

    @lru_cache(maxsize=3)
    def load_model(self, model_key):
        config = self.model_registry.get(model_key)
        if not config:
            raise ValueError(f"Model {model_key} not found in registry.")
        
        tokenizer = AutoTokenizer.from_pretrained(config["path"])
        
        if config["type"] == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(config["path"])
        elif config["type"] == "summarization":
            model = AutoModelForSeq2SeqLM.from_pretrained(config["path"])
        elif config["type"] == "generation":
            model = AutoModelForCausalLM.from_pretrained(config["path"])
        else:
            raise ValueError(f"Unsupported model type: {config['type']}")
        
        return tokenizer, model

    def get_pipeline(self, model_key):
        config = self.model_registry.get(model_key)
        if not config:
            raise ValueError(f"Model {model_key} not found in registry.")
        
        tokenizer, model = self.load_model(model_key)
        return pipeline(
            config["type"],
            model=model,
            tokenizer=tokenizer,
            device=self.device
        )
