
import streamlit as st
import pdfplumber
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pipeline import OptimizedPipeline

st.set_page_config(page_title="Medical NLP", layout="centered")
st.sidebar.title("🧠 Medical NLP App")
st.sidebar.markdown("Built by **Dr. Alelyani**") 
st.sidebar.markdown("📬 [LinkedIn](https://www.linkedin.com/in/alelyanit)")
st.title("📄 Medical NLP Pipeline")
uploaded_file = st.file_uploader("Upload a clinical PDF", type="pdf")

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    if not text.strip():
        st.warning("No readable text found in the uploaded PDF.")
    else:
        st.success("✅ PDF processed successfully.")
        pipeline = OptimizedPipeline()
        with st.spinner("🔄 Analyzing..."):
            result = pipeline.process_text(text)

        st.subheader("📌 Summary")
        st.write(result["summary"])

        st.subheader("🧬 PICO Elements")
        st.json(result["pico"])

        st.subheader("📈 Embedding Visualization (PCA)")
        try:
            emb = np.array(result["embeddings"]).reshape(1, -1)
            if emb.shape[0] < 2:
                st.warning("PCA requires at least 2 samples.")
            elif emb.shape[1] < 2:
                st.warning("Not enough embedding dimensions.")
            else:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(emb)
                fig, ax = plt.subplots()
                ax.scatter(reduced[:, 0], reduced[:, 1])
                ax.set_title("PCA Projection")
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"PCA error: {e}")
