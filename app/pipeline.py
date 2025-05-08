# pipeline.py
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import gc
from config.settings import Config

class OptimizedPipeline:
    def __init__(self):
        self.config = Config
        self._setup_device()
        self._load_models()
        self.process_count = 0
        
    def _setup_device(self):
        print(f"Initializing pipeline for device: {self.config.DEVICE}")
        if self.config.DEVICE == "mps":
            # Metal Performance Shaders (Apple Silicon)
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
    def _load_models(self):
        """Load models with memory optimization"""
        try:
            # Summarization model
            self.summarizer = pipeline(
                "summarization",
                model=self.config.SUMMARIZATION_MODEL,
                device=self.device,
                torch_dtype=torch.float32
            )
            
            # Embedding model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.EMBEDDING_MODEL)
            self.model = AutoModel.from_pretrained(
                self.config.EMBEDDING_MODEL
            ).to(self.device)
            
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
            
    def _cleanup(self):
        """Explicit memory cleanup"""
        gc.collect()
        if self.config.DEVICE == "mps":
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def process_text(self, text: str) -> dict:
        """Process text with memory constraints"""
        try:
            # Clean input text
            text = text[:self.config.MAX_INPUT_LENGTH]
            
            # Process in chunks if needed
            if len(text) > self.config.CHUNK_SIZE:
                return self._process_in_chunks(text)
                
            # Generate summary
            summary = self.summarizer(
                text,
                max_length=min(150, int(len(text)/4)),
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            
            # Generate embeddings
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy().flatten()
            
            # Simple PICO extraction (rule-based)
            pico = self._extract_pico(text)
            
            # Periodic cleanup
            self.process_count += 1
            if self.process_count % self.config.CLEANUP_EVERY == 0:
                self._cleanup()
                
            return {
                "summary": summary,
                "embeddings": embeddings,
                "pico": pico
            }
            
        except Exception as e:
            self._cleanup()
            print(f"Processing error: {e}")
            raise
            
    def _process_in_chunks(self, text: str) -> dict:
        """Process long text in manageable chunks"""
        # Implementation for chunked processing
        pass
        
    def _extract_pico(self, text: str) -> dict:
        """Simplified PICO extraction"""
        # Basic implementation - would expand with hybrid approach
        return {
            "participants": ["Sample participants info"],
            "intervention": ["Sample treatment"],
            "comparison": ["Sample control"],
            "outcome": ["Sample results"]
        }