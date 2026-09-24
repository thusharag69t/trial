# Central configuration for the Face Recognition Identification System.
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Directories ----
ENROLLED_IMAGES_DIR = os.path.join(BASE_DIR, "data", "enrolled_images")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "data", "test_images")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "face_database.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATIC_DIR = os.path.join(BASE_DIR, "static")
AUDIT_DB_PATH = os.path.join(DATABASE_DIR, "security_audit.sqlite3")
LEGACY_AUDIT_LOG_PATH = os.path.join(RESULTS_DIR, "security_audit_log.json")
SESSION_TTL_SECONDS = 60 * 60 * 12
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096

# ---- Face detection ----
# "hog" = fast, CPU-friendly (dlib HOG/SVM)
# "cnn" = more accurate on hard poses (dlib CNN)
# "auto" = automatically select available model (dlib if available, else OpenCV)
DETECTION_MODEL = "auto"
# Native-resolution detection avoids small false positives in textured backgrounds.
UPSAMPLE_TIMES = 1
# Haar detections clipped by the image boundary are commonly background false positives.
# Disable this when the camera frame is small or the face fills the frame.
REJECT_EDGE_TOUCHING_FACES = False


# ---- Matching ----
# Metric options: "cosine" (similarity in [0, 1]) or "euclidean" (L2 distance >= 0)
SIMILARITY_METRIC = "cosine"

# Default thresholds
# For Cosine Similarity: match accepted if similarity >= COSINE_THRESHOLD (e.g., 0.50)
COSINE_THRESHOLD = 0.50

# For Euclidean Distance: match accepted if distance <= EUCLIDEAN_THRESHOLD (e.g., 0.55)
EUCLIDEAN_THRESHOLD = 0.55

# Active threshold (dynamically set based on metric)
MATCH_THRESHOLD = COSINE_THRESHOLD if SIMILARITY_METRIC == "cosine" else EUCLIDEAN_THRESHOLD

