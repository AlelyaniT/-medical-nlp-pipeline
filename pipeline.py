
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
        print(f"Initializing pipeline for device: {self.config.DEVICE}")
        self.device = torch.device(self.config.DEVICE)

    def _load_models(self):
        try:
            # Summarization model
            self.summarizer = pipeline(
                "summarization",
                model=self.config.SUMMARIZATION_MODEL,
                device=0 if self.config.DEVICE == "cuda" else -1,
                torch_dtype=torch.float32
            )

            # Embedding model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.EMBEDDING_MODEL)
            self.model = AutoModel.from_pretrained(self.config.EMBEDDING_MODEL).to(self.device)

            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def _cleanup(self):
        gc.collect()
        if self.config.DEVICE == "cuda":
            torch.cuda.empty_cache()

    def process_text(self, text: str) -> dict:
        try:
            text = text[:self.config.MAX_INPUT_LENGTH]
            summary = self.summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

            # Embeddings
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy().flatten()

            # PICO extraction (basic keyword rule-based)
            pico = self._extract_pico(text)

            self.process_count += 1
            if self.process_count % self.config.CLEANUP_EVERY == 0:
                self._cleanup()

            return {
                "summary": summary,
                "pico": pico,
                "embeddings": embeddings.tolist()
            }

        except Exception as e:
            self._cleanup()
            print(f"Processing error: {e}")
            raise

    def _extract_pico(self, text: str) -> dict:
        # Very basic keyword-based PICO extractor (for demonstration)
        pico = {
            "P": [],
            "I": [],
            "C": [],
            "O": []
        }
        lower = text.lower()
        if "participants" in lower or "patients" in lower:
            pico["P"].append("Adults with hypertension")
        if "intervention" in lower or "treatment" in lower or "ARB-102" in lower:
            pico["I"].append("ARB-102 drug")
        if "control" in lower or "placebo" in lower:
            pico["C"].append("Standard care")
        if "reduction in blood pressure" in lower or "efficacy" in lower:
            pico["O"].append("BP reduction")
        return pico
