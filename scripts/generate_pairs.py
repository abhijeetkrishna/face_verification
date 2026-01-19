# ============================================================
# LFW FACE VERIFICATION PAIR GENERATOR (SKLEARN VERSION)
# ============================================================

import random
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_lfw_people

# -------------------------------
# CONFIG
# -------------------------------
DATA_HOME = "data/raw"
OUT_DIR = "data/processed"
MIN_FACES_PER_PERSON = 1
RESIZE = 0.4

POS_PAIRS_PER_ID = 5   # number of positive pairs per identity (if possible)
SEED = 42

# -------------------------------
# SETUP
# -------------------------------
random.seed(SEED)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# -------------------------------
# LOAD DATASET
# -------------------------------
lfw = fetch_lfw_people(
    min_faces_per_person=MIN_FACES_PER_PERSON,
    resize=RESIZE,
    data_home=DATA_HOME
)

images = lfw.images      # (N, H, W)
labels = lfw.target      # person IDs
names  = lfw.target_names

print(f"Images: {images.shape}")
print(f"Identities: {len(names)}")

# -------------------------------
# BUILD INDEX MAP
# -------------------------------
person_to_indices = {}
for idx, label in enumerate(labels):
    person_to_indices.setdefault(label, []).append(idx)

# -------------------------------
# GENERATE POSITIVE PAIRS
# -------------------------------
positive_pairs = []

for label, indices in person_to_indices.items():
    if len(indices) >= 2:
        for _ in range(POS_PAIRS_PER_ID):
            i1, i2 = random.sample(indices, 2)
            positive_pairs.append((i1, i2, 1))

print(f"Positive pairs: {len(positive_pairs)}")

# -------------------------------
# GENERATE NEGATIVE PAIRS
# -------------------------------
negative_pairs = []
all_labels = list(person_to_indices.keys())

while len(negative_pairs) < len(positive_pairs):
    l1, l2 = random.sample(all_labels, 2)
    i1 = random.choice(person_to_indices[l1])
    i2 = random.choice(person_to_indices[l2])
    negative_pairs.append((i1, i2, 0))

print(f"Negative pairs: {len(negative_pairs)}")

# -------------------------------
# COMBINE & SAVE
# -------------------------------
pairs = positive_pairs + negative_pairs
random.shuffle(pairs)

df = pd.DataFrame(pairs, columns=["idx1", "idx2", "same"])
out_path = Path(OUT_DIR) / "pairs.csv"
df.to_csv(out_path, index=False)

print(f"Total pairs: {len(df)}")
print(f"Saved to: {out_path.resolve()}")
