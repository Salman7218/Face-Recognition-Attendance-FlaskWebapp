"""Face-recognition based attendance system (Flask + OpenCV, browser camera)."""

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from datetime import date, datetime

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Biometric data lives in data/, which Flask does NOT serve (unlike static/).
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
MODEL_PATH = os.path.join(DATA_DIR, "face_model.yml")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# Tunables (override with environment variables).
# LBPH "confidence" is a distance: lower means more similar. Above this -> "Unknown".
RECOGNITION_THRESHOLD = float(os.environ.get("RECOGNITION_THRESHOLD", "70"))
# A person must be recognised in this many consecutive frames before attendance is marked.
CONFIRM_FRAMES = int(os.environ.get("CONFIRM_FRAMES", "8"))
# Safety limit for one attendance session in the browser.
SESSION_SECONDS = int(os.environ.get("SESSION_SECONDS", "300"))

NIMGS = 10                    # face images captured per new user
FRAME_EVERY = 2               # keep every Nth frame that contains a face
FACE_SIZE = (100, 100)
MIN_FACE = (80, 80)           # smallest face (pixels, in the uploaded frame) to accept
MAX_IMAGE_SIDE = 2000
STATE_TTL = 900               # seconds before idle sessions/registrations are dropped

# Letters/digits (any script), spaces, dots, apostrophes, hyphens. No underscore,
# because "_" separates name and ID in folder names.
NAME_RE = re.compile(r"[^\W_](?:[^\W_]|[ .'\-]){0,49}")
ROLL_RE = re.compile(r"\d{1,9}")
TOKEN_RE = re.compile(r"[A-Za-z0-9\-]{8,64}")

logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # reject oversized uploads

_face_detector = cv2.CascadeClassifier(CASCADE_PATH)
if _face_detector.empty():
    raise RuntimeError(f"Could not load Haar cascade from {CASCADE_PATH}")

_cv_lock = threading.Lock()            # OpenCV detector/recogniser are not thread-safe
_model_lock = threading.Lock()         # guards the cached recogniser
_attendance_lock = threading.RLock()   # guards attendance CSV read/write
_state_lock = threading.Lock()         # guards live sessions and pending registrations
_recognizer = None
_labels = {}                           # int label -> "Name_Roll" folder name
_live = {}                             # session token -> recognition streak state
_pending = {}                          # token -> registration in progress


def ensure_dirs():
    for path in (FACES_DIR, ATTENDANCE_DIR):
        os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------- attendance

def attendance_file():
    """Path of today's attendance CSV, created with a header if missing."""
    path = os.path.join(ATTENDANCE_DIR, f"Attendance-{date.today():%m_%d_%y}.csv")
    with _attendance_lock:
        if not os.path.isfile(path):
            os.makedirs(ATTENDANCE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("Name,Roll,Time\n")
    return path


def read_attendance():
    df = pd.read_csv(attendance_file(), dtype=str, keep_default_na=False)
    return df["Name"].tolist(), df["Roll"].tolist(), df["Time"].tolist()


def _ends_with_newline(path):
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


def add_attendance(person):
    """Record 'Name_Roll' for today unless already recorded. Returns True if written."""
    name, roll = person.rsplit("_", 1)
    now = datetime.now().strftime("%H:%M:%S")
    path = attendance_file()
    with _attendance_lock:
        _, rolls, _ = read_attendance()
        if roll in rolls:
            return False
        prefix = "\n" if os.path.getsize(path) > 0 and not _ends_with_newline(path) else ""
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{prefix}{name},{roll},{now}\n")
        return True


# ---------------------------------------------------------------------- users

def list_users():
    """Registered users as dicts with folder, name and roll."""
    users = []
    if not os.path.isdir(FACES_DIR):
        return users
    for folder in sorted(os.listdir(FACES_DIR)):
        if os.path.isdir(os.path.join(FACES_DIR, folder)) and "_" in folder:
            name, roll = folder.rsplit("_", 1)
            users.append({"folder": folder, "name": name, "roll": roll})
    return users


def display_name(folder):
    return folder.rsplit("_", 1)[0]


def validate_user(name, roll):
    """Return an error message for an invalid or duplicate user, else None."""
    if not NAME_RE.fullmatch(name):
        return ("Name must start with a letter or digit and may contain letters, "
                "digits, spaces, dots, apostrophes and hyphens (max 50 characters).")
    if not ROLL_RE.fullmatch(roll):
        return "ID must be a number of up to 9 digits."
    if any(u["roll"] == roll for u in list_users()):
        return f"ID {roll} is already registered."
    return None


# ------------------------------------------------------------ face processing

def decode_upload():
    """Decode the uploaded 'image' file into a BGR frame, or None if invalid."""
    upload = request.files.get("image")
    if upload is None:
        return None
    buf = np.frombuffer(upload.read(), np.uint8)
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or max(img.shape[:2]) > MAX_IMAGE_SIDE or min(img.shape[:2]) < 64:
        return None
    return img


def detect_faces(gray, min_size=MIN_FACE):
    """Return detected face boxes (x, y, w, h), largest first."""
    equalized = cv2.equalizeHist(gray)  # helps detection in poor lighting
    with _cv_lock:
        faces = _face_detector.detectMultiScale(
            equalized, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
    boxes = [tuple(int(v) for v in f) for f in faces]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def preprocess_face(gray_face):
    """Normalise a grayscale face crop so training and prediction match."""
    face = cv2.resize(gray_face, FACE_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)                  # compensate uneven lighting
    return cv2.GaussianBlur(face, (3, 3), 0)  # suppress sensor noise


def _new_recognizer():
    return cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


def train_model():
    """Rebuild the recogniser from every image in data/faces. Returns sample count."""
    global _recognizer, _labels
    samples, ids, labels = [], [], {}
    for user in list_users():
        user_dir = os.path.join(FACES_DIR, user["folder"])
        user_samples = []
        for img_name in sorted(os.listdir(user_dir)):
            img = cv2.imread(os.path.join(user_dir, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                user_samples.append(preprocess_face(img))
        if user_samples:
            label_id = len(labels)
            labels[label_id] = user["folder"]
            samples.extend(user_samples)
            ids.extend([label_id] * len(user_samples))

    with _model_lock:
        if not samples:
            _recognizer, _labels = None, {}
            for path in (MODEL_PATH, LABELS_PATH):
                if os.path.exists(path):
                    os.remove(path)
            return 0

        recognizer = _new_recognizer()
        recognizer.train(samples, np.array(ids, dtype=np.int32))
        tmp_model = MODEL_PATH + ".tmp.yml"
        recognizer.write(tmp_model)
        os.replace(tmp_model, MODEL_PATH)

        tmp_labels = LABELS_PATH + ".tmp"
        with open(tmp_labels, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in labels.items()}, f)
        os.replace(tmp_labels, LABELS_PATH)

        _recognizer, _labels = recognizer, labels
    return len(samples)


def get_recognizer():
    """Return (recogniser, labels), loading the saved model on first use."""
    global _recognizer, _labels
    with _model_lock:
        if _recognizer is None and os.path.isfile(MODEL_PATH) and os.path.isfile(LABELS_PATH):
            try:
                recognizer = _new_recognizer()
                recognizer.read(MODEL_PATH)
                with open(LABELS_PATH, encoding="utf-8") as f:
                    labels = {int(k): v for k, v in json.load(f).items()}
                _recognizer, _labels = recognizer, labels
            except (cv2.error, OSError, ValueError):
                logger.exception("Saved face model is unreadable; it will be rebuilt.")
                return None, {}
        return _recognizer, _labels


def identify_face(gray_face):
    """Return (folder_name or None, distance). None means unknown person."""
    recognizer, labels = get_recognizer()
    if recognizer is None:
        return None, None
    face = preprocess_face(gray_face)
    with _cv_lock:
        label, distance = recognizer.predict(face)
    if distance > RECOGNITION_THRESHOLD or label not in labels:
        return None, distance
    return labels[label], distance


# ------------------------------------------------------- per-session state

def valid_token(value):
    value = (value or "").strip()
    return value if TOKEN_RE.fullmatch(value) else None


def _prune(store, limit=500):
    """Drop idle entries and cap the store size. Call with _state_lock held."""
    now = time.monotonic()
    for key in [k for k, v in store.items() if now - v["seen"] > STATE_TTL]:
        del store[key]
    while len(store) >= limit:
        del store[min(store, key=lambda k: store[k]["seen"])]


def update_streak(token, folder):
    """Track consecutive matches for one browser session.

    Returns (streak, newly_confirmed). A person is confirmed once, after
    CONFIRM_FRAMES consecutive matching frames.
    """
    with _state_lock:
        state = _live.get(token)
        if state is None:
            _prune(_live)
            state = _live[token] = {"name": None, "streak": 0, "done": set(), "seen": 0.0}
        state["seen"] = time.monotonic()
        if folder is None:
            state["name"], state["streak"] = None, 0
            return 0, False
        if folder == state["name"]:
            state["streak"] += 1
        else:
            state["name"], state["streak"] = folder, 1
        if state["streak"] >= CONFIRM_FRAMES and folder not in state["done"]:
            state["done"].add(folder)
            return state["streak"], True
        return state["streak"], False


# ------------------------------------------------------------------- views

def render_home(mess=None, error=False):
    names, rolls, times = read_attendance()
    return render_template(
        "home.html", names=names, rolls=rolls, times=times, l=len(names),
        totalreg=len(list_users()), datetoday2=datetime.now().strftime("%d-%B-%Y"),
        mess=mess, error=error)


def render_users(mess=None, error=False):
    users = list_users()
    return render_template("listusers.html", users=users, totalreg=len(users),
                           mess=mess, error=error)


# ------------------------------------------------------------------- routes

@app.after_request
def add_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Permissions-Policy", "camera=(self)")
    return resp


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large."), 413


@app.errorhandler(500)
def server_error(_):
    if request.path.startswith("/api/"):
        return jsonify(error="Internal server error."), 500
    return "Internal server error.", 500


@app.route("/")
def home():
    return render_home()


@app.route("/listusers")
def listusers():
    return render_users()


@app.route("/deleteuser", methods=["POST"])
def deleteuser():
    folder = request.form.get("user", "")
    match = next((u for u in list_users() if u["folder"] == folder), None)
    if match is None:  # only exact registered folder names are accepted
        return render_users("User not found.", error=True), 404

    with _model_lock:
        shutil.rmtree(os.path.join(FACES_DIR, match["folder"]), ignore_errors=True)
    train_model()  # rebuilds the model, or removes it when no users remain
    return render_users(f"Deleted {match['name']}.")


@app.route("/start")
def start():
    return redirect("/attendance")


@app.route("/attendance")
def attendance():
    if get_recognizer()[0] is None:
        if not list_users():
            return render_home("There are no registered users. Please add a new user first.", True)
        train_model()
        if get_recognizer()[0] is None:
            return render_home("No usable face images found. Please re-register users.", True)
    return render_template("attendance.html", session_seconds=SESSION_SECONDS,
                           confirm_frames=CONFIRM_FRAMES)


@app.route("/add", methods=["POST"])
def add():
    name = request.form.get("newusername", "").strip()
    roll = request.form.get("newuserid", "").strip()
    error = validate_user(name, roll)
    if error:
        return render_home(error, True)
    return render_template("register.html", name=name, roll=roll, needed=NIMGS)


# ---------------------------------------------------------------- API: attendance

@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    token = valid_token(request.form.get("token"))
    if token is None:
        return jsonify(error="Invalid session."), 400
    frame = decode_upload()
    if frame is None:
        return jsonify(error="Invalid image."), 400
    if get_recognizer()[0] is None:
        return jsonify(error="No trained model. Please add a user first."), 409

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if not faces:
        update_streak(token, None)
        return jsonify(status="none", face=None)

    x, y, w, h = faces[0]  # only the closest/largest face
    box = {"x": x, "y": y, "w": w, "h": h}
    folder, _ = identify_face(gray[y:y + h, x:x + w])
    streak, confirmed = update_streak(token, folder)
    if folder is None:
        return jsonify(status="unknown", face=box)

    name = display_name(folder)
    if confirmed:
        written = add_attendance(folder)
        return jsonify(status="marked" if written else "already", face=box, name=name)
    if streak >= CONFIRM_FRAMES:
        return jsonify(status="done", face=box, name=name)
    return jsonify(status="confirming", face=box, name=name,
                   progress=streak, needed=CONFIRM_FRAMES)


# ---------------------------------------------------------------- API: registration

@app.route("/api/register/start", methods=["POST"])
def api_register_start():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    roll = str(data.get("roll", "")).strip()
    error = validate_user(name, roll)
    if error:
        return jsonify(error=error), 400

    with _state_lock:
        _prune(_pending)
        for key in [k for k, p in _pending.items() if p["roll"] == roll]:
            if _pending[key]["name"] != name:
                return jsonify(error=f"ID {roll} is being registered right now."), 409
            del _pending[key]  # same person retrying: drop the abandoned attempt
        token = uuid.uuid4().hex
        _pending[token] = {"name": name, "roll": roll, "crops": [],
                           "detected": 0, "seen": time.monotonic()}
    return jsonify(token=token, needed=NIMGS)


@app.route("/api/register/frame", methods=["POST"])
def api_register_frame():
    token = valid_token(request.form.get("token"))
    with _state_lock:
        reg = _pending.get(token) if token else None
        if reg is not None:
            reg["seen"] = time.monotonic()
    if reg is None:
        return jsonify(error="Registration expired. Please start again."), 404
    frame = decode_upload()
    if frame is None:
        return jsonify(error="Invalid image."), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    box = None
    with _state_lock:
        if faces:
            x, y, w, h = faces[0]
            box = {"x": x, "y": y, "w": w, "h": h}
            reg["detected"] += 1
            if reg["detected"] % FRAME_EVERY == 0 and len(reg["crops"]) < NIMGS:
                reg["crops"].append(gray[y:y + h, x:x + w].copy())
        captured = len(reg["crops"])
    return jsonify(face=box, captured=captured, needed=NIMGS, done=captured >= NIMGS)


@app.route("/api/register/finish", methods=["POST"])
def api_register_finish():
    data = request.get_json(silent=True) or {}
    token = valid_token(str(data.get("token", "")))
    with _state_lock:
        reg = _pending.pop(token, None) if token else None
    if reg is None:
        return jsonify(error="Registration expired. Please start again."), 404
    if len(reg["crops"]) < NIMGS:
        return jsonify(error="Not enough face images were captured."), 400
    error = validate_user(reg["name"], reg["roll"])
    if error:
        return jsonify(error=error), 409

    user_dir = os.path.join(FACES_DIR, f"{reg['name']}_{reg['roll']}")
    created = False
    try:
        os.makedirs(user_dir, exist_ok=False)
        created = True
        for crop in reg["crops"]:
            if not cv2.imwrite(os.path.join(user_dir, f"{uuid.uuid4().hex}.jpg"), crop):
                raise OSError("Could not save a face image.")
        train_model()
    except (OSError, cv2.error):
        logger.exception("Registration failed")
        if created:
            shutil.rmtree(user_dir, ignore_errors=True)
            try:
                train_model()
            except (OSError, cv2.error):
                logger.exception("Could not restore the face model")
        return jsonify(error="Could not save the new user. Please try again."), 500
    return jsonify(ok=True, message=f"Added {reg['name']} with {len(reg['crops'])} face images.")


@app.route("/api/register/cancel", methods=["POST"])
def api_register_cancel():
    data = request.get_json(silent=True) or {}
    token = valid_token(str(data.get("token", "")))
    with _state_lock:
        if token:
            _pending.pop(token, None)
    return jsonify(ok=True)


ensure_dirs()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
