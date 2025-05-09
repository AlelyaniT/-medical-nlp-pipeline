
import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_MODEL = "distilbert-base-uncased"
    SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
    MAX_INPUT_LENGTH = 2000
    CHUNK_SIZE = 512
    CLEANUP_EVERY = 3
