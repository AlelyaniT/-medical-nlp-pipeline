import streamlit as st
import pdfplumber
from pipeline import OptimizedPipeline

# ------------------------------
# Sidebar - Branding & Contact
# ------------------------------
st.sidebar.title("🔬 Medical NLP App")
st.sidebar.markdown("Built by **Dr. Alelyani**")
st.sidebar.markdown("📧 Contact: [LinkedIn](https://www.linkedin.com/in/alelyanit)")

st.sidebar.markdown("---")
st.sidebar.markdown("Upload a clinical PDF to extract:")
st.sidebar.markdown("- 🔍 **Summarization**")
st.sidebar.markdown("- 🧬 **PICO elements**")
st.sidebar.markdown("- 📈 **Embeddings**")

# ------------------------------
# Main Interface
# ------------------------------
st.title("📄 Medical NLP Pipeline")
st.markdown("Upload a clinical PDF to extract insights using AI-powered NLP tools.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    if len(text.strip()) == 0:
        st.warning("No readable text found in PDF.")
    else:
        st.success("✅ PDF loaded successfully.")
        st.text_area("📄 Extracted Text Preview", value=text[:2000], height=200)

        pipeline = OptimizedPipeline()
        with st.spinner("🔄 Processing text..."):
            result = pipeline.process_text(text)

        st.subheader("📌 Summary")
        st.write(result["summary"])

        st.subheader("📌 PICO Extraction")
        st.json(result["pico"])

        st.subheader("📌 Embeddings Vector Length")
        st.write(len(result["embeddings"]))
