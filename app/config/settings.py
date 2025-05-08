import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_MODEL = "distilbert-base-uncased"
    SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
    MAX_INPUT_LENGTH = 10000
    CHUNK_SIZE = 1000
    CLEANUP_EVERY = 3
