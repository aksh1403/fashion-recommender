import json
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors

from recommender import build_model, extract_features

DATA_DIR = Path("data")
N_RECOMMENDATIONS = 5

st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("Fashion Recommender System")


@st.cache_resource
def load_model():
    return build_model()


@st.cache_resource
def load_index():
    emb_path = DATA_DIR / "embeddings.npy"
    names_path = DATA_DIR / "filenames.json"
    if not emb_path.exists() or not names_path.exists():
        return None

    embeddings = np.load(emb_path)
    with open(names_path) as f:
        filenames = json.load(f)

    nn = NearestNeighbors(
        n_neighbors=min(N_RECOMMENDATIONS, len(filenames)),
        algorithm="brute",
        metric="euclidean",
    ).fit(embeddings)
    return nn, filenames


index = load_index()
if index is None:
    st.error("Embeddings not found. Run `python extract_features.py` first.")
    st.stop()

nn, filenames = index
model = load_model()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        features = extract_features(uploaded_file, model)
    except Exception as e:
        st.error(f"Could not process the image: {e}")
        st.stop()

    uploaded_file.seek(0)
    st.subheader("Your image")
    st.image(uploaded_file, width=300)

    _, indices = nn.kneighbors([features])

    st.subheader("Similar items")
    cols = st.columns(len(indices[0]))
    for col, idx in zip(cols, indices[0]):
        with col:
            st.image(filenames[idx])