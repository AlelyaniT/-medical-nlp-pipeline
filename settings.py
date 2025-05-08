
import torch

class Config:
    # Device settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Model selection
    EMBEDDING_MODEL = "distilbert-base-uncased"
    SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

    # Memory and text handling
    MAX_INPUT_LENGTH = 2000
    CHUNK_SIZE = 512

    # Cleanup
    CLEANUP_EVERY = 3
