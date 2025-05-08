import streamlit as st
import pdfplumber
from app.pipeline import OptimizedPipeline

st.title("📄 Medical NLP Pipeline")
st.markdown("Upload a clinical PDF to extract Summary, PICO elements, and Embeddings.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    if len(text.strip()) == 0:
        st.warning("No readable text found in PDF.")
    else:
        st.success("✅ PDF loaded successfully.")
        st.text_area("Extracted Text Preview", value=text[:2000], height=200)

        pipeline = OptimizedPipeline()
        with st.spinner("Processing text..."):
            result = pipeline.process_text(text)

        st.subheader("📌 Summary")
        st.write(result["summary"])

        st.subheader("📌 PICO Extraction")
        st.json(result["pico"])

        st.subheader("📌 Embeddings Length")
        st.write(len(result["embeddings"]))
