# Fashion Recommender System

Upload a photo of a clothing item and get the 5 most visually similar items from a dataset of 44,000+ fashion product images.

Images are converted into 2048-dimensional embeddings with a pretrained **ResNet50** (ImageNet weights), and a **k-nearest-neighbours** search over those embeddings returns the closest matches. The interface is built with **Streamlit**.

<!-- Add a screenshot once the app is running:
![Demo](docs/demo.png)
-->

## How it works

1. **Feature extraction.** Every dataset image is resized to 224x224 and passed through ResNet50 (classification head removed) followed by global max pooling. The result is a 2048-dimensional vector per image.
2. **Normalisation.** Each vector is L2-normalised, so Euclidean distance reflects visual similarity.
3. **Storage.** All embeddings are saved to `data/embeddings.npy`, with matching image paths in `data/filenames.json`.
4. **Recommendation.** When you upload an image, it is embedded the same way, and a brute-force k-nearest-neighbours search (Euclidean distance) returns the 5 closest dataset images.

```
Dataset images -> ResNet50 -> 2048-d embeddings -> saved to disk
                                                        |
Uploaded image -> ResNet50 -> embedding -> kNN search ->+-> 5 similar items
```

## Project structure

```
fashion-recommender/
├── app.py                # Streamlit web app
├── extract_features.py   # Builds embeddings for the dataset
├── recommender.py        # Shared model and feature extraction code
├── requirements.txt
├── README.md
├── .gitignore
├── images/               # Dataset goes here (not committed)
└── data/                 # Generated embeddings (not committed)
```

## Requirements

- Python **3.10, 3.11 or 3.12** (TensorFlow does not yet support the newest Python releases)
- An internet connection on first run, to download the ResNet50 weights (~90 MB)

## Setup

```bash
git clone https://github.com/aksh1403/fashion-recommender.git
cd fashion-recommender
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is not included in this repository. Download one and place the images in the `images/` folder. This project was built with the
[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset from Kaggle (~44k images).

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`.

For a quick test, 50 to 100 images are enough.

## Usage

**1. Generate the embeddings** (one time; slow on CPU for large datasets):

```bash
python extract_features.py
```

This creates `data/embeddings.npy` and `data/filenames.json`.

**2. Launch the app:**

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload an image, and the 5 most similar items appear below it.

The first page load takes a while because the model and embeddings are loaded once and then cached. Later uploads take a few seconds at most.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No images found in 'images/'` | The folder is empty or the files have unsupported extensions. |
| `Embeddings not found` in the app | Run `python extract_features.py` first. |
| `ModuleNotFoundError` | The virtual environment is not active, or dependencies are not installed. |
| `No matching distribution found for tensorflow` | Your Python version is too new. Use Python 3.10 to 3.12. |
| Blank page or very slow start | Run `streamlit run app.py --server.headless true --server.fileWatcherType none`. Avoid keeping the project inside a synced folder such as OneDrive. |
| Broken images in the results | Run the app from the project root so the `images/...` paths resolve. |

## Tech stack

TensorFlow / Keras, scikit-learn, Streamlit, Pillow, NumPy

## Future improvements

- Faster search with an approximate index such as FAISS for larger datasets
- Filtering results by category, colour or gender using the dataset metadata
- Fine-tuning the model on fashion data for more accurate matches
- Deploying a hosted demo (Streamlit Community Cloud or Hugging Face Spaces)

