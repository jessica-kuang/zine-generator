import numpy as np
import anthropic
import base64
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# need to make src a package
from schema import TasteProfile, VisualIdentity

# load embeddings
def load_embeddings(embeddings_dir: str, handle:str):
    embeddings_path = Path(embeddings_dir) / f"{handle}_embeddings.npy"
    paths_path = Path(embeddings_dir) / f"{handle}_image_paths.txt"

    embeddings = np.load(embeddings_path)
    image_paths = paths_path.read_text().strip().split("\n")

    print(f"loaded {len(image_paths)} embeddings")
    return embeddings, image_paths

# elbow method
def find_optimal_k(embeddings, k_range=range(2,7)):
    inertias = []
    silhouette_scores = []

    for k in k_range:
        print(f"trying k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)
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

# cluster images
def cluster_images(embeddings, image_paths, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for cluster_id in range(k):
        # get indices of images in this cluster
        indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[indices]

        # find centroid image
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        centroid_idx = indices[np.argmin(distances)]

        clusters[cluster_id] = {
            "image_paths": [image_paths[i] for i in indices],
            "centroid_image": image_paths[centroid_idx],
            "size": len(indices)
        }
    return clusters

# encode image for claude
def encode_image(image_path: str):
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

# label cluster with claude vision 
def label_cluster(image_path: str, client: anthropic.Anthropic):
    image_data = encode_image(image_path)

    #detect image type
    suffix = Path(image_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }
    media_type = media_type_map.get(suffix, "image/jpeg")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": """You are an expert in fashion, aesthetics, and visual culture.
                        Analyze this image and respond with ONLY a JSON object in this exact format:
                        {
                            "archetype": "2-3 word aesthetic label (e.g. dark romantic, quiet luxury)",
                            "motifs": ["motif1", "motif2", "motif3"],
                            "texture_language": ["texture1", "texture2"],
                            "mood": "one sentence describing the overall mood"
                        }
                        No other text, just the JSON."""
                    }
                ]
            }
        ]
    )
    response_text = message.content[0].text.strip()
    return json.loads(response_text)

# build visual identity
def build_visual_identity(clusters, client):
    archetypes = []
    all_motifs = []
    all_textures = []
    labels = {}

    for cluster_id, cluster_data in clusters.items():
        print(f"labeling cluster {cluster_id} ({cluster_data['size']} images)...")
        label = label_cluster(cluster_data["centroid_image"], client)
        labels[cluster_id] = label

        archetypes.append(label["archetype"])
        all_motifs.extend(label["motifs"])
        all_textures.extend(label["texture_language"])

    # deduplicate
    unique_motifs = list(dict.fromkeys(all_motifs))[:6]
    unique_textures = list(dict.fromkeys(all_textures))[:4]

    return archetypes, unique_motifs, unique_textures, labels

# main
if __name__ == "__main__":
    client = anthropic.Anthropic()

    # load
    embeddings, image_paths = load_embeddings("data/embeddings", "jessica")

    # find k
    optimal_k = min(find_optimal_k(embeddings), 4)

    # cluster
    clusters = cluster_images(embeddings, image_paths, optimal_k)

    # label with claude
    archetypes, motifs, textures, labels = build_visual_identity(clusters, client)

    print("\n--- visual identity ---")
    print(f"archetypes: {archetypes}")
    print(f"motifs: {motifs}")
    print(f"textures: {textures}")

    # save labels
    output_path = Path("data/profiles/cluster_labels.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"clusters": clusters, "labels": {str(k): v for k, v in labels.items()}}, f, indent=2)

    print(f"\nsaved cluster labels to {output_path}")