import numpy as np
from keras import Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
from PIL import Image

IMG_SIZE = (224, 224)


def build_model():
    """ResNet50 (ImageNet weights, no top) + global max pooling -> 2048-d vector."""
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    return Sequential([base, GlobalMaxPooling2D()])

def load_image(source):
    """Load a path or file-like object as a 224x224 RGB float array."""
    img = Image.open(source).convert("RGB").resize(IMG_SIZE)
    return np.asarray(img, dtype=np.float32)


def extract_features(source, model):
    """Return an L2-normalised feature vector for a single image."""
    arr = preprocess_input(np.expand_dims(load_image(source), axis=0))
    result = model.predict(arr, verbose=0).flatten()
    return result / np.linalg.norm(result)