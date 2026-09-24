"""
FastAPI REST API Backend Server for Face Recognition Identification System.

Provides endpoints for:
  - System status & database inspection
  - Enrolling new face images
  - Query face identification with annotated image overlays
  - Identity management & enrolled photos inspection / lightbox
  - Threshold evaluation sweep metrics and curves
  - Config management
  - Web UI static asset hosting
"""
import os
import sys
import io
import base64
import glob
import hashlib
import json
import secrets
import sqlite3
import time
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone
import cv2
import numpy as np
from fastapi import Cookie, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.database import FaceDatabase
from src.face_utils import load_image_from_bytes, load_image
from src.enroll import enroll_single_image
from src.identify import identify_image, draw_annotations, encode_image_base64
from src.evaluate import run_evaluation

app = FastAPI(
    title="Face Recognition System API",
    description="REST API for enrolling, identifying, and evaluating face recognition metrics.",
    version="1.1.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global database instance
db = FaceDatabase(config.DATABASE_PATH)


class ConfigUpdateModel(BaseModel):
    metric: Optional[str] = None
    threshold: Optional[float] = None


class AuthModel(BaseModel):
    email: str
    password: str


class SignupModel(AuthModel):
    name: str


login_attempts = defaultdict(deque)
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_ATTEMPTS = 5


def audit_connection():
    connection = sqlite3.connect(config.AUDIT_DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_audit_database():
    os.makedirs(config.DATABASE_DIR, exist_ok=True)
    with audit_connection() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY, name TEXT NOT NULL, salt BLOB NOT NULL,
                password_hash BLOB NOT NULL, created_at TEXT NOT NULL, is_admin INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY, email TEXT NOT NULL REFERENCES users(email), expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL, event TEXT NOT NULL,
                actor TEXT NOT NULL, resource TEXT NOT NULL, outcome TEXT NOT NULL, severity TEXT NOT NULL,
                timestamp TEXT NOT NULL, request_id TEXT, client_ip TEXT, user_agent TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_category ON audit_events(category);
            CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
        if "is_admin" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        if connection.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1").fetchone()[0] == 0:
            connection.execute("UPDATE users SET is_admin = 1 WHERE email = (SELECT email FROM users ORDER BY created_at LIMIT 1)")
        settings = connection.execute("SELECT name, value FROM settings").fetchall()
        existing_events = connection.execute("SELECT COUNT(*) AS total FROM audit_events").fetchone()["total"]
        if existing_events == 0 and os.path.exists(config.LEGACY_AUDIT_LOG_PATH):
            try:
                with open(config.LEGACY_AUDIT_LOG_PATH, "r", encoding="utf-8") as legacy_file:
                    legacy_events = json.load(legacy_file)
                for event in legacy_events if isinstance(legacy_events, list) else []:
                    connection.execute(
                        """INSERT INTO audit_events
                        (category,event,actor,resource,outcome,severity,timestamp,request_id,client_ip,user_agent)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (event.get("category", "security"), event.get("event", "Legacy event"),
                         event.get("actor", "Unknown"), event.get("resource", "Unknown"),
                         event.get("outcome", "Recorded"), event.get("severity", "info"),
                         event.get("timestamp", datetime.now(timezone.utc).isoformat()), None, None, None),
                    )
                os.replace(config.LEGACY_AUDIT_LOG_PATH, f"{config.LEGACY_AUDIT_LOG_PATH}.migrated")
            except (OSError, json.JSONDecodeError):
                pass
    for setting in settings:
        if setting["name"] == "metric" and setting["value"] in ("cosine", "euclidean"):
            config.SIMILARITY_METRIC = setting["value"]
        elif setting["name"] == "threshold":
            config.MATCH_THRESHOLD = float(setting["value"])
            if config.SIMILARITY_METRIC == "cosine": config.COSINE_THRESHOLD = config.MATCH_THRESHOLD
            else: config.EUCLIDEAN_THRESHOLD = config.MATCH_THRESHOLD


def password_digest(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240000)


def create_session(email):
    token = secrets.token_urlsafe(32)
    with audit_connection() as connection:
        connection.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        connection.execute("INSERT INTO sessions(token_hash,email,expires_at) VALUES (?, ?, ?)",
                           (hashlib.sha256(token.encode()).hexdigest(), email, time.time() + config.SESSION_TTL_SECONDS))
    return token


def require_user(authorization: Optional[str] = Header(None), session_token: Optional[str] = Cookie(None, alias="intelliface_session")):
    token = authorization[7:].strip() if authorization and authorization.startswith("Bearer ") else session_token
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with audit_connection() as connection:
        user = connection.execute(
            """SELECT users.email, users.name, users.is_admin FROM sessions JOIN users ON users.email = sessions.email
               WHERE sessions.token_hash = ? AND sessions.expires_at > ?""", (token_hash, time.time())
        ).fetchone()
    if not user: raise HTTPException(status_code=401, detail="Session expired or invalid.")
    return dict(user)


def client_key(request):
    return f"{request.client.host if request.client else 'unknown'}:{request.headers.get('X-Forwarded-For', '')}"


def check_login_rate_limit(request, email):
    key = f"{client_key(request)}:{email}"
    now = time.time()
    attempts = login_attempts[key]
    while attempts and attempts[0] <= now - LOGIN_WINDOW_SECONDS:
        attempts.popleft()
    if len(attempts) >= LOGIN_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")


def record_failed_login(request, email):
    login_attempts[f"{client_key(request)}:{email}"].append(time.time())


async def read_image_upload(upload):
    data = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image upload is too large.")
    image = load_image_from_bytes(bytes(data))
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")
    height, width = image.shape[:2]
    if max(height, width) > config.MAX_IMAGE_DIMENSION:
        scale = config.MAX_IMAGE_DIMENSION / max(height, width)
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return image


def safe_photo_path(name, filename):
    if not name or Path(name).name != name or name in {".", ".."} or not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid photo path.")
    root = Path(config.ENROLLED_IMAGES_DIR).resolve()
    target = (root / name / filename).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid photo path.") from exc
    return str(target)


def safe_identity_dir(name):
    if not name or Path(name).name != name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid identity path.")
    root = Path(config.ENROLLED_IMAGES_DIR).resolve()
    target = (root / name).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid identity path.") from exc
    return str(target)


def require_admin(user):
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Administrator access required.")


def write_audit_event(event, actor, resource, outcome, category="security", severity="info", request=None):
    try:
        with audit_connection() as connection:
            connection.execute(
                """INSERT INTO audit_events
                (category,event,actor,resource,outcome,severity,timestamp,request_id,client_ip,user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (category, event, actor, resource, outcome, severity, datetime.now(timezone.utc).isoformat(),
                 request.headers.get("X-Request-ID") if request else None,
                 request.client.host if request and request.client else None,
                 request.headers.get("User-Agent") if request else None),
            )
    except sqlite3.Error:
        pass


def read_audit_events(category="all", limit=100):
    with audit_connection() as connection:
        query = "SELECT * FROM audit_events ORDER BY id DESC LIMIT ?" if category == "all" else "SELECT * FROM audit_events WHERE category = ? ORDER BY id DESC LIMIT ?"
        rows = connection.execute(query, (limit,) if category == "all" else (category, limit)).fetchall()
    return [dict(row) for row in rows]


initialize_audit_database()


@app.get("/api/status")
def get_system_status(user=Depends(require_user)):
    people = db.list_people()
    details = db.get_identity_details()
    total_embeddings = sum(details.values())
    return {
        "status": "online",
        "database_path": config.DATABASE_PATH,
        "enrolled_identities": len(people),
        "total_embeddings": total_embeddings,
        "identities": details,
        "detection_model": config.DETECTION_MODEL,
        "metric": config.SIMILARITY_METRIC,
        "threshold": config.MATCH_THRESHOLD,
    }


@app.post("/api/auth/signup")
def signup(request: Request, response: Response, data: SignupModel):
    email = data.email.strip().lower()
    if len(data.password) < 8 or not data.name.strip():
        raise HTTPException(status_code=400, detail="Name and an 8-character password are required.")
    salt = secrets.token_bytes(16)
    try:
        with audit_connection() as connection:
            is_admin = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
            connection.execute("INSERT INTO users(email,name,salt,password_hash,created_at,is_admin) VALUES (?, ?, ?, ?, ?, ?)",
                               (email, data.name.strip(), salt, password_digest(data.password, salt), datetime.now(timezone.utc).isoformat(), is_admin))
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    write_audit_event("Account created", data.name.strip(), email, "Completed", request=request)
    response.set_cookie("intelliface_session", create_session(email), httponly=True, samesite="lax", max_age=config.SESSION_TTL_SECONDS)
    return {"user": {"email": email, "name": data.name.strip()}}


@app.post("/api/auth/login")
def login(request: Request, response: Response, data: AuthModel):
    email = data.email.strip().lower()
    check_login_rate_limit(request, email)
    with audit_connection() as connection:
        user = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not user or not secrets.compare_digest(password_digest(data.password, user["salt"]), user["password_hash"]):
        record_failed_login(request, email)
        write_audit_event("Login failed", email, "Workspace", "Rejected", severity="danger", request=request)
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    write_audit_event("Login succeeded", user["name"], "Workspace", "Authenticated", request=request)
    response.set_cookie("intelliface_session", create_session(email), httponly=True, samesite="lax", max_age=config.SESSION_TTL_SECONDS)
    return {"user": {"email": email, "name": user["name"]}}


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie("intelliface_session")
    return {"success": True}


@app.post("/api/enroll")
async def enroll_face(
    request: Request,
    name: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(require_user),
):
    name = name.strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Person name cannot be empty.")

    try:
        image_rgb = await read_image_upload(file)
    except HTTPException as exc:
        write_audit_event("Enrollment failed", user["name"], name, exc.detail, category="enrollment", severity="danger", request=request)
        raise

    success = enroll_single_image(name, image_rgb, db, save_file=True)
    if not success:
        write_audit_event("Enrollment failed", user["name"], name, "No usable face", category="enrollment", severity="danger", request=request)
        raise HTTPException(
            status_code=422,
            detail=(
                f"No usable face detected for '{name}'. Keep one face centered, "
                "move closer, improve lighting, and capture again."
            ),
        )

    details = db.get_identity_details()
    write_audit_event("Identity enrolled", user["name"], name, "Completed", category="enrollment", request=request)
    return {
        "success": True,
        "message": f"Successfully enrolled face for '{name}'.",
        "person_name": name,
        "total_photos_for_person": details.get(name, 1),
        "total_identities": len(db.list_people()),
    }


@app.post("/api/identify")
async def identify_face(
    request: Request,
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    metric: Optional[str] = Form(None),
    user=Depends(require_user),
):
    try:
        image_rgb = await read_image_upload(file)
    except HTTPException as exc:
        write_audit_event("Recognition failed", user["name"], file.filename or "unnamed upload", exc.detail, category="recognition", severity="danger", request=request)
        raise

    use_metric = metric if metric in ("cosine", "euclidean") else config.SIMILARITY_METRIC
    use_threshold = threshold if threshold is not None else (
        config.COSINE_THRESHOLD if use_metric == "cosine" else config.EUCLIDEAN_THRESHOLD
    )
    max_threshold = 1.0 if use_metric == "cosine" else 2.0
    if not 0.0 <= use_threshold <= max_threshold:
        raise HTTPException(status_code=422, detail=f"Threshold must be between 0 and {max_threshold}.")

    results = identify_image(
        image_rgb,
        db=db,
        threshold=use_threshold,
        metric=use_metric
    )

    annotated_rgb = draw_annotations(image_rgb, results)
    b64_image = encode_image_base64(annotated_rgb)
    known_faces = ", ".join(result["name"] for result in results if result.get("is_known")) or "Unknown faces"
    write_audit_event("Face recognition", user["name"], known_faces, f"{len(results)} face(s) detected",
                      category="recognition", severity="success" if results else "warning", request=request)

    return {
        "face_count": len(results),
        "threshold_used": use_threshold,
        "metric_used": use_metric,
        "faces": results,
        "annotated_image_base64": b64_image,
    }


@app.get("/api/identities")
def list_identities(user=Depends(require_user)):
    people = db.list_people()
    details = db.get_identity_details()

    identities_list = []
    for name in people:
        person_dir = safe_identity_dir(name)
        photo_count = details.get(name, 0)
        first_photo_b64 = ""

        if os.path.exists(person_dir):
            files = [
                f for f in sorted(os.listdir(person_dir))
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            if files:
                first_path = os.path.join(person_dir, files[0])
                first_img = load_image(first_path)
                if first_img is not None:
                    first_photo_b64 = encode_image_base64(first_img)

        identities_list.append({
            "name": name,
            "embedding_count": photo_count,
            "thumbnail_base64": first_photo_b64,
        })

    return {"identities": identities_list}


@app.get("/api/identities/{name}/photos")
def get_identity_photos(name: str, user=Depends(require_user)):
    person_dir = safe_identity_dir(name)
    if not os.path.exists(person_dir):
        return {"name": name, "photos": []}

    photos = []
    files = [
        f for f in sorted(os.listdir(person_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    for fname in files:
        fpath = os.path.join(person_dir, fname)
        img_rgb = load_image(fpath)
        if img_rgb is not None:
            b64 = encode_image_base64(img_rgb)
            stat = os.stat(fpath)
            photos.append({
                "filename": fname,
                "size_kb": round(stat.st_size / 1024, 1),
                "url": f"/api/identities/{name}/photos/{fname}",
                "base64": b64,
            })

    return {"name": name, "count": len(photos), "photos": photos}


@app.get("/api/identities/{name}/photos/{filename}")
def serve_identity_photo(name: str, filename: str, user=Depends(require_user)):
    fpath = safe_photo_path(name, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Photo file not found.")
    return FileResponse(fpath)


@app.delete("/api/identities/{name}")
def delete_identity(request: Request, name: str, user=Depends(require_user)):
    removed = db.remove_person(name)
    person_dir = safe_identity_dir(name)
    if os.path.exists(person_dir):
        import shutil
        shutil.rmtree(person_dir, ignore_errors=True)

    if not removed and not os.path.exists(person_dir):
        write_audit_event("Identity deletion failed", user["name"], name, "Not found", severity="danger", request=request)
        raise HTTPException(status_code=404, detail=f"Identity '{name}' not found.")
    write_audit_event("Identity deleted", user["name"], name, "Removed", severity="warning", request=request)
    return {"success": True, "message": f"Deleted identity '{name}' and all associated photos."}


@app.delete("/api/identities/{name}/photos/{filename}")
def delete_identity_photo(request: Request, name: str, filename: str, user=Depends(require_user)):
    fpath = safe_photo_path(name, filename)
    if os.path.exists(fpath):
        os.remove(fpath)
        write_audit_event("Photo deleted", user["name"], f"{name}/{filename}", "Removed", severity="warning", request=request)
        return {"success": True, "message": f"Deleted photo '{filename}' for identity '{name}'."}
    write_audit_event("Photo deletion failed", user["name"], f"{name}/{filename}", "Not found", severity="danger", request=request)
    raise HTTPException(status_code=404, detail="Photo file not found.")


@app.get("/api/evaluate")
def run_or_get_evaluation(metric: Optional[str] = Query("cosine"), user=Depends(require_user)):
    try:
        eval_res = run_evaluation(metric=metric)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    plot_path = os.path.join(config.RESULTS_DIR, "threshold_curve.png")

    plot_b64 = ""
    if os.path.exists(plot_path):
        with open(plot_path, "rb") as f:
            plot_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "summary": eval_res,
        "plot_base64": plot_b64,
    }


@app.post("/api/config")
def update_config(request: Request, data: ConfigUpdateModel, user=Depends(require_user)):
    effective_metric = data.metric if data.metric in ("cosine", "euclidean") else config.SIMILARITY_METRIC
    if data.metric in ("cosine", "euclidean"):
        config.SIMILARITY_METRIC = data.metric
    if data.threshold is not None:
        max_threshold = 1.0 if effective_metric == "cosine" else 2.0
        if not 0.0 <= data.threshold <= max_threshold:
            raise HTTPException(status_code=422, detail=f"Threshold must be between 0 and {max_threshold}.")
        config.MATCH_THRESHOLD = float(data.threshold)
        if config.SIMILARITY_METRIC == "cosine":
            config.COSINE_THRESHOLD = float(data.threshold)
        else:
            config.EUCLIDEAN_THRESHOLD = float(data.threshold)

    with audit_connection() as connection:
        connection.executemany("INSERT INTO settings(name,value) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value = excluded.value",
                               [("metric", config.SIMILARITY_METRIC), ("threshold", str(config.MATCH_THRESHOLD))])
    write_audit_event("Configuration changed", user["name"], "Recognition settings",
                      f"{config.SIMILARITY_METRIC} / {config.MATCH_THRESHOLD:.2f}", request=request)
    return {
        "success": True,
        "metric": config.SIMILARITY_METRIC,
        "threshold": config.MATCH_THRESHOLD,
    }


@app.get("/api/audit-log")
def get_audit_log(category: Optional[str] = Query("all"), limit: int = Query(100, ge=1, le=500), user=Depends(require_user)):
    events = read_audit_events(category or "all", limit)
    return {"events": events, "total": len(events)}


@app.delete("/api/audit-log")
def clear_audit_log(request: Request, user=Depends(require_user)):
    require_admin(user)
    with audit_connection() as connection:
        connection.execute("DELETE FROM audit_events")
    write_audit_event("Audit log cleared", user["name"], "Security audit trail", "Cleared", severity="warning", request=request)
    return {"success": True, "message": "Security audit log cleared."}


# Redirect root to login page
from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/login.html")


# Serve static web frontend
os.makedirs(config.STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("Starting Face Recognition API Server on http://localhost:8000...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
