import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path

# device set up
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# load model
def load_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess

# load images
def load_images(upload_dir: str):
    upload_path = Path(upload_dir)
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [
        p for p in upload_path.iterdir()
        if p.suffix.lower() in supported
    ]
    return image_paths

# embed images
def embed_images(image_paths, model, preprocess, device):
    embeddings = []
    valid_paths = []

    for path in image_paths:
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.cpu().numpy())
                valid_paths.append(str(path))
        except Exception as e:
            print(f"skipping {path.name}: {e}")
    return np.vstack(embeddings), valid_paths

# save embeddings
def save_embeddings(embeddings, image_paths, output_dir: str, handle: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / f"{handle}_embeddings.npy", embeddings)

    paths_file = output_path / f"{handle}_image_paths.txt"
    paths_file.write_text("\n".join(image_paths))
    print(f"saved {len(image_paths)} embeddings to {output_path}")

# main
if __name__ == "__main__":
    device = get_device()
    print(f"using device: {device}")

    model, preprocess = load_clip_model(device)
    print("clip model loaded")

    image_paths = load_images("data/uploads")
    print(f"found {len(image_paths)} images")

    embeddings, valid_paths = embed_images(image_paths, model, preprocess, device)
    save_embeddings(embeddings, valid_paths, "data/embeddings", "jessica")