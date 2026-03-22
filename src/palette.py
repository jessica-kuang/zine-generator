import numpy as np
import anthropic
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
import json

# same pixels from image
def sample_pixel(image_path: str, n_samples: int = 1000):
    img = Image.open(image_path).convert("RBG")
    pixels = np.array(img).reshape(-1, 3)
    indices = np.random.choice(len(pixels), min(n_samples, len(pixels)), replace=False)
    return pixels[indices]