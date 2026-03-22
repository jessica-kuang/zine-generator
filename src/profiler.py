import numpy as np
import anthropic
import base64
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from src.schema import TasteProfile, VisualIdentity

# load embeddings
def load_embeddings(embeddings_dir: str, handle:str):
    embeddings_path = Path(embeddings_dir) / f"{handle}_embeddings.npy"
    paths_path = Path(embeddings_dir) / f"{handle}_image_paths.txt"

    embeddings = np.load(embeddings_path)
    image_paths = paths_path.read_text().strip.split("\n")

    print(f"loaded {len(image_paths)} embeddings")
    return embeddings, image_paths

# elbow method
def find_optimal_k(embeddings, k_range=range(2,7)):
    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.intertia_)
        silhouette_scores.append(silhouette_score(embeddings, labels))

        # plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        ax1.plot(list(k_range), inertias, 'bo-')
        ax1.set_xlabel('k')
        ax1.set_ylabel('inertia')
        ax1.set_title('elbow method')

        ax2.plot(list(k_range), silhouette_scores, 'ro-')
        ax2.set_xlabel('k')
        ax2.set_ylabel('silhouette score')
        ax2.set_title('silhouette scores')

        plt.tight_layout()
        plt.savefig('data/profiles/elbow_curve.png')
        plt.close()

        # pick k with highest silhouette score
        optimal_k = list(k_range)[np.argmax(silhouette_scores)]
        print(f"optimal k: {optimal_k}")
        return optimal_k

