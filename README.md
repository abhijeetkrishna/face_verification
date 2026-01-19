# Face Verification Service

## Problem Description

Modern photo applications (e.g. Google Photos) automatically group images by person without requiring predefined identities.
At the core of such systems is **face verification**: given two face images, decide whether they belong to the same person.

This project implements a **face verification service** that can later be used as a plug-and-play module in a larger photo organization system. The service takes two face images as input and performs a **binary classification** (`same person` / `different person`).

The project focuses on **verification**, not closed-set face recognition, as verification is the fundamental building block for scalable, open-set photo libraries.

In short, the goal is to do the following task

```
Given (face_A, face_B) → same person or not
```
---

## Dataset

We use the **Labeled Faces in the Wild (LFW)** dataset:

* Collection of face images of public figures
* Faces are already detected and aligned
* Standard benchmark for face verification

***Downloading the dataset***:

Dataset was taken from scikit-learn and then processed to generate matching and non-matching pairs. 

```
from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4,
    data_home="data/raw"
)
```

To reproduce the data needed for training, run the following commands:

```
python scripts/download_data.py
python scripts/generate_pairs.py
```

---

## Approach

### Embedding Extraction

Simple image comparison will fail for face verification because aspects lighting conditions, expressions, etc can make two images of the same person very different. Extracting embedding using a model like FaceNet will convert our 2D images into a 512-dimensional vector. Each image is now a point in this 512D space. This model is pre-trained to encode the identity of the face in the image. Hence images of the same person in this space will be close together.

* A **pretrained FaceNet model** is used to convert each face image into a fixed-length embedding vector.
* The embedding model is kept **frozen** to keep training fast and reproducible.

To extract the embedding run the following commands :

```
python scripts/extract_embeddings.py
```

### Models

The embeddings are used to train two models:

1. **Logistic Regression on Embedding Differences**
2. **Support Vector Machine**

The models are fine-tuned and both perform equally well.

---

## Evaluation

Models are evaluated using:

* Accuracy
* ROC AUC
* Confusion matrix

Similarity score distributions for positive and negative pairs are analyzed as part of EDA.

---

## Service

The trained verification model is exposed via a **Flask API**.

### Endpoint

`POST /verify`

**Request**

```json
{
  "image1": "<base64-encoded image>",
  "image2": "<base64-encoded image>"
}
```

**Response**

```json
{
  "same_person": true,
  "score": 0.87
}
```

---

## Project Structure

```
.
├── data/               # LFW data and processed embeddings
├── models/             # Trained models and thresholds
├── src/
│   ├── process_dataset.py
│   ├── extract_embeddings.py
│   ├── train_models.py
│   └── evaluate.py
├── service/
│   └── app.py          # Flask application
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Running the Project

### Environment setup

```bash
python -m venv faceEnv
source faceEnv/bin/activate #Mac
faceEnv\Scripts\activate #Windows
pip install -r requirements.txt
```

### Training

```bash
python scripts/download_data.py
python scripts/extract_embeddings.py
python scripts/train_models.py
python scripts/evaluate.py
```

### Run service

```bash
python service/app.py
```

---

## Containerization

Build and run with Docker:

```bash
docker build -t face-verification .
docker run -p 9696:9696 face-verification
```

---

## Future Work

* Face clustering to automatically build “people” groups
* Incremental updates as new photos are added
* Active learning for cluster refinement
* Cloud deployment and scalable embedding search (FAISS)

