# 🧠 Medical NLP Pipeline

This Streamlit app enables fast and interactive analysis of clinical documents using AI-powered NLP tools. Upload a PDF, and the app will extract a summary, identify PICO elements, and visualize text embeddings.

---

## 🚀 Features

- 📄 PDF Upload and Text Extraction
- 🔍 Abstractive Summarization
- 🧬 PICO Element Extraction (Participants, Intervention, Comparison, Outcome)
- 📈 Embedding Visualization using PCA
- 📥 Download Summary & PICO Report
- 🧠 Model Selector (BART, T5, DistilBERT)
- 📍 Branded Sidebar with Contact Info

---

## 📦 How to Use

1. **Clone the repo or upload it to Streamlit Cloud**
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run locally:
   ```bash
   streamlit run app.py
   ```

Or deploy on [Streamlit Cloud](https://streamlit.io/cloud) with `app.py` as the entry point.

---

## 📂 Project Structure

```
├── app.py
├── pipeline.py
├── model_manager.py
├── settings.py
├── requirements.txt
```

---

## 🧑‍💻 Author

Built by [Dr. Alelyani](https://www.linkedin.com/in/alelyanit)

---

## 📄 Sample Output

Try it with any clinical study PDF and explore:
- Extracted summary
- PICO table
- 2D visualization of the text’s semantic embedding
