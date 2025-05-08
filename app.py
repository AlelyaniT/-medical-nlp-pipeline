import streamlit as st
import pdfplumber
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from io import StringIO
from pipeline import OptimizedPipeline
from model_manager import ModelManager

# Sidebar: Branding & Info
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/512px-React-icon.svg.png", width=80)
st.sidebar.title("🧠 Medical NLP App")
st.sidebar.markdown("Built by **Dr. Alelyani**")
st.sidebar.markdown("📧 [Contact on LinkedIn](https://www.linkedin.com/in/turki-alelyani)")
st.sidebar.markdown("---")
st.sidebar.markdown("Upload a clinical PDF to extract:")
st.sidebar.markdown("- 🔍 **Summarization**")
st.sidebar.markdown("- 🧬 **PICO elements**")
st.sidebar.markdown("- 📈 **Embeddings**")
st.sidebar.markdown("---")
st.sidebar.markdown("ℹ️ This app uses clinical NLP models to process and visualize medical research.")

# Main UI
st.title("📄 Medical NLP Pipeline")

# Model selection (placeholder logic)
model_name = st.selectbox("Choose summarization model", ["sshleifer/distilbart-cnn-12-6", "facebook/bart-large-cnn", "t5-base"])

uploaded_file = st.file_uploader("Upload a clinical PDF", type="pdf")

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    if not text.strip():
        st.warning("No readable text found in the uploaded PDF.")
    else:
        st.success("✅ PDF processed successfully.")
        st.text_area("📄 Extracted Text", text[:2000], height=200)

        pipeline = OptimizedPipeline()
        with st.spinner("🔄 Analyzing..."):
            result = pipeline.process_text(text)

        # Summary
        st.subheader("📌 Summary")
        st.write(result["summary"])

        # PICO
        st.subheader("🧬 PICO Elements")
        st.json(result["pico"])

        # Embeddings Visualization (safe PCA)
        st.subheader("📈 Embedding Visualization (PCA)")
        try:
            emb = np.array(result["embeddings"]).reshape(1, -1)
            if emb.shape[0] < 2:
                st.warning("PCA requires at least 2 samples for projection.")
            elif emb.shape[1] < 2:
                st.warning("Not enough embedding dimensions to project.")
            else:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(emb)
                fig, ax = plt.subplots()
                ax.scatter(reduced[:, 0], reduced[:, 1], c='blue')
                ax.set_title("PCA Projection")
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ Unable to project embeddings: {e}")

        # Downloadable report
        st.subheader("📄 Download Report")
        report = f"Summary:\n{result['summary']}\n\nPICO:\n{result['pico']}"
        st.download_button(
            label="Download summary & PICO",
            data=report,
            file_name="nlp_report.txt",
            mime="text/plain"
        )
