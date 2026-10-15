import json
from pathlib import Path

import numpy as np
from keras.applications.resnet50 import preprocess_input
from tqdm import tqdm

from recommender import build_model, load_image

IMAGE_DIR = Path("images")
OUT_DIR = Path("data")
BATCH_SIZE = 32
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    if not IMAGE_DIR.is_dir():
        raise SystemExit(f"Folder '{IMAGE_DIR}/' not found. Create it and add your dataset.")

    paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found in '{IMAGE_DIR}/'. Add your dataset first.")

    OUT_DIR.mkdir(exist_ok=True)
    model = build_model()

    features, kept = [], []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Extracting features"):
        imgs, ok = [], []
        for p in paths[i:i + BATCH_SIZE]:
            try:
                imgs.append(load_image(p))
                ok.append(p)
            except Exception as e:
                print(f"Skipping {p}: {e}")
        if not imgs:
            continue

        out = model.predict(preprocess_input(np.stack(imgs)), verbose=0)
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
        features.append(out)
        kept.extend(p.as_posix() for p in ok)

    if not features:
        raise SystemExit("No images could be processed.")

    np.save(OUT_DIR / "embeddings.npy", np.vstack(features))
    with open(OUT_DIR / "filenames.json", "w") as f:
        json.dump(kept, f)
    print(f"Saved {len(kept)} embeddings to '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()