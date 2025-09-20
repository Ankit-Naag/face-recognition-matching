import io
import os
import ast
import json
import runpod
import base64
import logging
import tempfile
import traceback
import psycopg2
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageOps

# -------- InsightFace imports --------
from functools import lru_cache
from insightface.app import FaceAnalysis
from typing import Dict, Any, Tuple, List

try:
    connection = psycopg2.connect(
        database="db",
        user="user",
        password="password",
        host="host",
        port="5432"
    )
    cursor = connection.cursor()
except Exception as e:
    logging.error(f"Error connecting to database: {e}")

# =======================
# Router
# =======================
def handler(event):
    input_data = event.get("input", {})
    action = input_data.get("type", "")
    data = input_data.get("data", {})

    if action == "embedding":
        return face_embedding_handler(data)              # DeepFace embeddings
    elif action == "match":
        return face_match_handler(data)                  # DeepFace matching
    elif action == "insight_embedding":
        return insightface_embedding_handler(data)       # InsightFace embeddings
    elif action == "insight_match":
        return insightface_match_handler(data)           # InsightFace matching
    else:
        return {"status": False, "message": "Invalid action"}


# =======================
# Utilities
# =======================
def _decode_image_to_pil(image_base64: str) -> Image.Image:
    image_data = base64.b64decode(image_base64)
    if not image_data:
        raise ValueError("image is required")
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    return image

def _to_float_list(x):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return None
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

# =======================
# DeepFace
# =======================
def face_match_handler(event):
    try:
        image_base64 = event.get("image_base64")
        face_database = event.get("face_database", [])

        logging.info(f"face_database sample: {json.dumps(face_database[:1], default=str)}")

        image = _decode_image_to_pil(image_base64)

        temp_image_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
                image.save(temp_image_file, format="JPEG")
                temp_image_path = temp_image_file.name

            face_data = DeepFace.represent(
                img_path=temp_image_path,
                model_name="VGG-Face",
                enforce_detection=False,
                detector_backend="retinaface"
            )
            logging.info(f"face_data sample: {json.dumps(face_data[:1], default=str)}")
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

        match_stats = {}  # { qr_code_id: { count, total_distance, best_match, best_distance } }

        for face in face_data:
            face_encoding = face["embedding"]
            for existing in face_database:
                existing_embedding = existing.get("face_embedding")
                qr_code_id = existing.get("qr_code_id")
                if not qr_code_id or existing_embedding is None:
                    continue

                existing_embedding = _to_float_list(existing_embedding)
                if existing_embedding is None:
                    logging.error(f"Invalid embedding for qr_code_id={qr_code_id}")
                    continue

                try:
                    res = DeepFace.verify(existing_embedding, face_encoding,
                                          model_name="VGG-Face",
                                          detector_backend="retinaface",
                                          silent=True)
                except Exception:
                    logging.error(json.dumps({
                        "message": "DeepFace.verify error",
                        "qr_code_id": qr_code_id,
                        "error": traceback.format_exc()
                    }))
                    continue

                if res.get("verified"):
                    distance = res.get("distance", float("inf"))
                    if qr_code_id not in match_stats:
                        match_stats[qr_code_id] = {
                            "count": 1,
                            "total_distance": distance,
                            "best_match": existing,
                            "best_distance": distance
                        }
                    else:
                        match_stats[qr_code_id]["count"] += 1
                        match_stats[qr_code_id]["total_distance"] += distance
                        if distance < match_stats[qr_code_id]["best_distance"]:
                            match_stats[qr_code_id]["best_match"] = existing
                            match_stats[qr_code_id]["best_distance"] = distance

        if not match_stats:
            logging.info("No match found")
            return {
                "status": True,
                "match": None,
                "face_data": face_data,
                "message": "No match found"
            }

        best_qr_code_id = min(
            match_stats.items(),
            key=lambda x: x[1]["total_distance"] / x[1]["count"]
        )[0]
        best_match_qr_code = min(
            match_stats.items(),
            key=lambda x: x[1]["best_distance"]
        )[0]
        if best_qr_code_id != best_match_qr_code:
            if match_stats[best_match_qr_code]["count"] >= match_stats[best_qr_code_id]["count"]:
                best_qr_code_id = best_match_qr_code

        return {
            "status": True,
            "match": None,
            "face_data": None,
            "match": match_stats[best_qr_code_id]["best_match"],
            "face_data": face_data
        }

    except Exception as e:
        logging.error(f"face_match_handler error: {str(e)}")
        return {"status": False, "message": str(e)}


def face_embedding_handler(event):
    try:
        image_base64 = event.get("image_base64")
        image = _decode_image_to_pil(image_base64)
        face_data = []

        temp_image_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
                image.save(temp_image_file, format="JPEG")
                temp_image_path = temp_image_file.name

            face_data = DeepFace.represent(
                img_path=temp_image_path,
                model_name="VGG-Face",
                enforce_detection=False,
                detector_backend="retinaface"
            )
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

        return {"status": True, "face_data": face_data}

    except Exception as e:
        logging.error(f"face_embedding_handler error: {str(e)}")
        return {"status": False, "message": str(e)}


# =======================
# InsightFace
# =======================

@lru_cache(maxsize=1)
def _get_insight_app(ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    return rgb[:, :, ::-1]


def insightface_embedding_handler(event) -> Dict[str, Any]:
    try:
        image_base64 = event.get("image_base64")
        min_detect_score = float(event.get("min_detect_score", 0.50))
        use_cpu = bool(event.get("use_cpu", True))
        det_w = int(event.get("det_w", 640))
        det_h = int(event.get("det_h", 640))

        image = _decode_image_to_pil(image_base64)
        bgr = _pil_to_bgr(image)

        ctx_id = -1 if use_cpu else 0
        app = _get_insight_app(ctx_id=ctx_id, det_size=(det_w, det_h))

        faces = [f for f in app.get(bgr) if getattr(f, "det_score", 0.0) >= min_detect_score]

        face_data = [{"embedding": _l2_normalize(f.normed_embedding).tolist()} for f in faces]

        return {"status": True, "face_data": face_data}

    except Exception as e:
        logging.error(f"insightface_embedding_handler error: {str(e)}")
        return {"status": False, "message": str(e)}


def insightface_match_handler(event) -> Dict[str, Any]:
    try:
        image_base64 = event.get("image_base64")

        similarity_threshold = float(event.get("similarity_threshold", 0.50))
        min_detect_score = float(event.get("min_detect_score", 0.50))
        use_cpu = bool(event.get("use_cpu", True))
        det_size = (int(event.get("det_w", 640)), int(event.get("det_h", 640)))

        # Decode & prepare
        image = _decode_image_to_pil(image_base64)
        bgr = _pil_to_bgr(image)
        ctx_id = -1 if use_cpu else 0
        app = _get_insight_app(ctx_id=ctx_id, det_size=det_size)

        # Detect & embed query faces
        faces = [f for f in app.get(bgr) if getattr(f, "det_score", 0.0) >= min_detect_score]
        if not faces:
            return {"status": True, "match": None, "face_data": [], "message": "No faces detected"}

        # Prepare query embeddings (ArcFace iresnet100, L2-normalized)
        query_faces = [{"embedding": _l2_normalize(f.normed_embedding)} for f in faces]

        face_database = []
        try:
            cursor.execute("""
                SELECT fi.id, fi.face_embedding, qi.qr_code_id, qal.email
                FROM api_faceinfo fi
                JOIN api_qrinfo qi ON fi.qr_code_id = qi.id
                LEFT JOIN api_qraccesslog qal ON qi.id = qal.qr_code_id
                WHERE fi.face_embedding != '[]'
            """)
            rows = cursor.fetchall()
            for row in rows:
                face_id, face_embedding, qr_code_id, email = row
                try:
                    # Parse JSON face_embedding if stored as string
                    if isinstance(face_embedding, str):
                        face_embedding = json.loads(face_embedding)
                    face_database.append({
                        "id": face_id,
                        "face_embedding": face_embedding,
                        "qr_code_id": qr_code_id,
                        "email": email if email else ""
                    })
                except json.JSONDecodeError:
                    logging.warning(f"Invalid face_embedding for face_id {face_id}")
                    continue
        except Exception as e:
            logging.error(f"Database query error: {str(e)}")
            return {"status": False, "message": f"Database query error: {str(e)}"}

        # Prepare DB as qr_code_id -> [embeddings]
        per_person: Dict[str, List[np.ndarray]] = {}
        for rec in face_database:
            pid = rec.get("qr_code_id")
            emb = _to_float_list(rec.get("face_embedding"))
            if not pid or emb is None:
                continue
            emb = _l2_normalize(np.asarray(emb, dtype=np.float32))
            per_person.setdefault(pid, []).append(emb)

        if not per_person:
            return {"status": True, "match": None, "face_data": [], "message": "Empty or invalid face_database"}

        # --- Aggregation identical in spirit to your DeepFace version, but using cosine distance = 1 - sim ---
        # match_stats = { pid: {count, total_distance, best_match, best_distance} }
        match_stats: Dict[str, Dict[str, Any]] = {}

        for q in query_faces:
            q_emb = q["embedding"]
            for pid, emb_list in per_person.items():
                # Best similarity for this person against this query face
                if not emb_list:
                    continue
                pid_sim = max(float(np.dot(q_emb, db_emb)) for db_emb in emb_list)
                if pid_sim < similarity_threshold:
                    # Not 'verified' -> skip (this mirrors DeepFace.verify gating)
                    continue

                distance = 1.0 - pid_sim  # cosine distance in [0..2]; with L2-norm, practical range ~[0..2], often [0..1]
                stats = match_stats.setdefault(pid, {
                    "count": 0,
                    "total_distance": 0.0,
                    "best_match": None,
                    "best_distance": float("inf")
                })
                stats["count"] += 1
                stats["total_distance"] += distance
                if distance < stats["best_distance"]:
                    # Pick any representative record for this pid
                    stats["best_distance"] = distance
                    stats["best_match"] = next((r for r in face_database if r.get("qr_code_id") == pid), None)

        if not match_stats:
            return {"status": True, "match": None, "face_data": [{"embedding": q["embedding"].tolist()} for q in query_faces], "message": "No match found"}

        # Choose by lowest average distance
        def avg_dist(item):
            _pid, s = item
            return s["total_distance"] / max(s["count"], 1)

        best_qr_code_id = min(match_stats.items(), key=avg_dist)[0]
        best_match_qr_code = min(match_stats.items(), key=lambda x: x[1]["best_distance"])[0]

        # Tiebreaker exactly like your code
        if best_qr_code_id != best_match_qr_code:
            if match_stats[best_match_qr_code]["count"] >= match_stats[best_qr_code_id]["count"]:
                best_qr_code_id = best_match_qr_code

        return {
            "status": True,
            "match": match_stats[best_qr_code_id]["best_match"],
            # keep output compatible with your saver
            "face_data": [{"embedding": q["embedding"].tolist()} for q in query_faces]
        }

    except Exception as e:
        logging.exception("insightface_match_handler error")
        return {"status": False, "message": str(e)}


# =======================
# Start RunPod serverless
# =======================
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
