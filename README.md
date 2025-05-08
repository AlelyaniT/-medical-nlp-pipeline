# 🧠 Medical NLP Pipeline

This project extracts clinical summaries, PICO elements, and embeddings from uploaded PDF documents.

## Features
- Abstractive summarization
- Embedding generation using DistilBERT
- Simplified rule-based PICO extraction
- Streamlit web interface

## Usage
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Upload a clinical PDF and view results

## Folder Structure
```
app/
  ├── pipeline.py
  ├── model_manager.py
  └── config/
      └── settings.py
app.py
requirements.txt
README.md
```
