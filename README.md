# 🧑‍💻 Face Recognition & Matching API

This project provides a **serverless face recognition and matching API** built with:

- [DeepFace](https://github.com/serengil/deepface) (VGG-Face embeddings & matching)
- [InsightFace](https://github.com/deepinsight/insightface) (ArcFace embeddings & cosine similarity)
- [RunPod](https://www.runpod.io/) serverless deployment
- PostgreSQL as a face/QR database backend

The API supports **embedding generation** and **face matching** with multiple backends for flexibility.

---

## 🚀 Features
- Generate face embeddings using **DeepFace (VGG-Face)** or **InsightFace (ArcFace)**.
- Match faces against a database of embeddings.
- Normalize embeddings for cosine similarity scoring.
- PostgreSQL integration for storing and querying face embeddings.
- Deployable on **RunPod serverless** for scalability.
- Works with **Base64 encoded images** (JPEG/PNG).

---

## 🛠️ Tech Stack
- **Python 3.9+**
- DeepFace
- InsightFace
- NumPy
- Pillow (PIL)
- psycopg2 (PostgreSQL client)
- RunPod SDK

---

## 📂 Project Structure
```

├── app.py                # Main serverless handler
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation

````

---

## ⚡ API Endpoints

The entry point is the `handler(event)` function.  
The event payload must include `input.type` and `input.data`.

### 1. DeepFace Embedding
Generate embeddings using VGG-Face.

```json
{
  "input": {
    "type": "embedding",
    "data": {
      "image_base64": "<base64-encoded-image>"
    }
  }
}
````

**Response:**

```json
{
  "status": true,
  "face_data": [
    {
      "embedding": [ ...vector... ]
    }
  ]
}
```

---

### 2. DeepFace Match

Match against a face database.

```json
{
  "input": {
    "type": "match",
    "data": {
      "image_base64": "<base64-encoded-image>",
      "face_database": [
        {
          "qr_code_id": "123",
          "face_embedding": [ ...vector... ]
        }
      ]
    }
  }
}
```

**Response:**

```json
{
  "status": true,
  "match": {
    "qr_code_id": "123",
    "face_embedding": [ ...vector... ]
  },
  "face_data": [ ... ]
}
```

---

### 3. InsightFace Embedding

Generate ArcFace embeddings.

```json
{
  "input": {
    "type": "insight_embedding",
    "data": {
      "image_base64": "<base64-encoded-image>"
    }
  }
}
```

**Response:**

```json
{
  "status": true,
  "face_data": [
    {
      "embedding": [ ...vector... ]
    }
  ]
}
```

---

### 4. InsightFace Match (with DB integration)

Matches face(s) against embeddings stored in PostgreSQL.

```json
{
  "input": {
    "type": "insight_match",
    "data": {
      "image_base64": "<base64-encoded-image>",
      "similarity_threshold": 0.6
    }
  }
}
```

**Response:**

```json
{
  "status": true,
  "match": {
    "qr_code_id": "456",
    "email": "user@example.com",
    "face_embedding": [ ... ]
  },
  "face_data": [
    { "embedding": [ ... ] }
  ]
}
```

---

## ⚙️ Setup & Installation

1. **Clone repository**

```bash
git clone https://github.com/ankit-naag/face-recognition-matching.git
cd face-recognition-matching
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set environment variables**
   Update your PostgreSQL credentials inside the code:

```python
connection = psycopg2.connect(
    database="db",
    user="user",
    password="password",
    host="host",
    port="5432"
)
```

5. **Run locally**

```bash
python app.py
```

---

## 📦 Deployment (RunPod Serverless)

This project is designed for **RunPod serverless GPU deployment**.
Start the handler with:

```bash
runpod.serverless.start({"handler": handler})
```

Then deploy using the RunPod dashboard or CLI.