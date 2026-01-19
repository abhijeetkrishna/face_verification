# ============================================================
# Face Verification API
# ============================================================

import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import joblib
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1


# -------------------------------
# CONFIG
# -------------------------------
DEVICE = "cpu"  # change to "cuda" if available
MODEL_DIR = Path("model")

THRESHOLD = 0.5  # decision threshold on LR probability


# -------------------------------
# LOAD MODELS (ONCE)
# -------------------------------
print("Loading FaceNet...")
embedding_model = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)

print("Loading verification model...")
verifier = joblib.load(MODEL_DIR / "logistic_regression.joblib")

print("Models loaded.")


# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def decode_image(base64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def extract_embedding(img: Image.Image) -> np.ndarray:
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = embedding_model(img_t)
    return emb.cpu().numpy()[0]


# -------------------------------
# FEATURE CONSTRUCTION
# -------------------------------
def build_features(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    abs_diff = np.abs(e1 - e2)
    prod = e1 * e2
    cos_sim = np.sum(e1 * e2, keepdims=True)
    l2_dist = np.linalg.norm(e1 - e2, keepdims=True)
    return np.hstack([abs_diff, prod, cos_sim, l2_dist]).reshape(1, -1)


# -------------------------------
# FLASK APP
# -------------------------------
app = Flask("faceVerification")


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json()

    if not data or "image1" not in data or "image2" not in data:
        return jsonify({"error": "Expected fields: image1, image2"}), 400

    try:
        img1 = decode_image(data["image1"])
        img2 = decode_image(data["image2"])

        emb1 = extract_embedding(img1)
        emb2 = extract_embedding(img2)

        X = build_features(emb1, emb2)

        prob = float(verifier.predict_proba(X)[0, 1])
        decision = prob >= THRESHOLD

        return jsonify({
            "same_person": decision,
            "score": prob,
            "threshold": THRESHOLD
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
