# ============================================================
# FACE EMBEDDINGS FROM SKLEARN LFW (FACENET)
# ============================================================

import torch
import numpy as np
from sklearn.datasets import fetch_lfw_people
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm

# -------------------------------
# DEVICE
# -------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

# -------------------------------
# LOAD LFW (SKLEARN)
# -------------------------------
lfw = fetch_lfw_people(
    min_faces_per_person=1,
    resize=0.4,
    data_home="data/raw"
)

images = lfw.images  # (N, H, W) grayscale
print("Images:", images.shape)

# -------------------------------
# TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),               # auto-detect uint8
    transforms.Resize((160, 160)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),                 # -> [0, 1]
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# -------------------------------
# MODEL
# -------------------------------
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# -------------------------------
# EMBEDDING EXTRACTION
# -------------------------------
embeddings = []

with torch.no_grad():
    for img in tqdm(images):

        img = transform((img * 255).astype(np.uint8))

        img = img.unsqueeze(0).to(device)

        emb = model(img)
        embeddings.append(emb.cpu().numpy()[0])

embeddings = np.vstack(embeddings)

# -------------------------------
# SAVE
# -------------------------------
np.save("data/processed/embeddings.npy", embeddings)

print("Embeddings saved:", embeddings.shape)
