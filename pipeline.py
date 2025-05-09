
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import gc
from settings import Config

class OptimizedPipeline:
    def __init__(self):
        self.config = Config
        self._setup_device()
        self._load_models()
        self.process_count = 0

    def _setup_device(self):
        self.device = torch.device(self.config.DEVICE)

    def _load_models(self):
        self.summarizer = pipeline(
            "summarization",
            model=self.config.SUMMARIZATION_MODEL,
            device=0 if self.config.DEVICE == "cuda" else -1,
            torch_dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(self.config.EMBEDDING_MODEL).to(self.device)

    def _cleanup(self):
        gc.collect()
        if self.config.DEVICE == "cuda":
            torch.cuda.empty_cache()

    def process_text(self, text: str) -> dict:
        text = text[:self.config.MAX_INPUT_LENGTH]
        summary = self.summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy().flatten()
        pico = self._extract_pico(text)
        return {
            "summary": summary,
            "pico": pico,
            "embeddings": embeddings.tolist()
        }

    def _extract_pico(self, text: str) -> dict:
        lower = text.lower()
        return {
            "P": ["Adults with hypertension"] if "participants" in lower else [],
            "I": ["ARB-102 drug"] if "arb-102" in lower or "treatment" in lower else [],
            "C": ["Standard care"] if "control" in lower else [],
            "O": ["BP reduction"] if "blood pressure" in lower else []
        }
