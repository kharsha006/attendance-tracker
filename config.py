# =============================================================
#  config.py  —  All settings for the attendance system
#  Phase 3: Multi-Camera Tracking + ReID + Activity Recognition
# =============================================================

# ---------------------------------------------------------------
# ATTENDANCE CAMERAS
# These cameras are used for presence-based logging (first_seen/last_seen).
# ---------------------------------------------------------------
GLOBAL_FRAME_BUFFER = {}

ATTENDANCE_CAMERA_URLS = [1]

# ---------------------------------------------------------------
OFFICE_CAMERA_A_URL = 0
# OFFICE_CAMERA_A_URL = "rtsp://user:pass@192.168.1.11/stream1"
# OFFICE_CAMERA_B_URL = "rtsp://user:pass@192.168.1.12/stream1"

# ---------------------------------------------------------------
# ALL FOUR CAMERAS — unified config for the tracking engine
# name:     short identifier used in DB + dashboard
# url:      RTSP or integer (webcam)
# role:     'entry' | 'exit' | 'office'
# ---------------------------------------------------------------
CAMERA_CONFIG = [
    {"name": "camera_1", "url": OFFICE_CAMERA_A_URL, "role": "office"},
]

# ---------------------------------------------------------------
# OFFICE HOURS
# ---------------------------------------------------------------
OFFICE_START_HOUR   = 9
OFFICE_START_MINUTE = 0
OFFICE_END_HOUR     = 21
OFFICE_END_MINUTE   = 0

# ---------------------------------------------------------------
# FRAME SAMPLING INTERVALS (seconds)
# ---------------------------------------------------------------
FRAME_INTERVAL_SECONDS          = 1.0    # Changed from 0.2 to prevent GPU OOM crashes
TRACKING_FRAME_INTERVAL_SECONDS = 0.5   # office tracking cameras (faster)
ACTIVITY_FRAME_INTERVAL_SECONDS = 10   # legacy alias kept for compat
ACTIVITY_SAMPLE_DURATION_SECONDS = ACTIVITY_FRAME_INTERVAL_SECONDS

# ---------------------------------------------------------------
# INFERENCE DEVICE  (Forced to CPU per user request)
# ---------------------------------------------------------------
USE_GPU = False

def get_torch_device() -> str:
    """Return 'cpu'."""
    return "cpu"

def get_yolo_device():
    """Device argument for Ultralytics YOLO ('cpu')."""
    return "cpu"

def get_onnx_providers() -> list:
    """ONNX Runtime provider chain for InsightFace."""
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------
# YOLO  (person detection model)
# Options: yolov8n.pt (fastest) | yolov8s.pt | yolov8m.pt | yolov8l.pt | yolov8x.pt (most accurate)
# ---------------------------------------------------------------
YOLO_MODEL_NAME = "yolov8n.pt"

# ---------------------------------------------------------------
# FACE RECOGNITION  (InsightFace — buffalo_l model)
# ---------------------------------------------------------------
FACE_RECOGNITION_MODEL  = "buffalo_l"   # InsightFace model pack name
CONFIDENCE_THRESHOLD    = 0.48          # minimum cosine similarity for a match (0.48 prevents false positives)
MIN_FACE_SIZE           = 35            # min face width/height in px (ignores far away blurry faces until they get closer)
FACE_DETECTION_INTERVAL = 5             # run face recognition every N frames
                                        # (tracking covers the rest)

# RetinaFace detector tuning — lower values detect more / smaller / farther faces
FACE_DET_SIZE           = (960, 960)    # detection input size (640=fast, 960=better multi-person)
FACE_DET_THRESH         = 0.35          # RetinaFace confidence threshold
FACE_MIN_DET_SCORE      = 0.35          # post-filter in camera_processor (must be <= FACE_DET_THRESH)

# YOLO finds every person in frame; InsightFace runs on each person crop for distant faces
USE_YOLO_PERSON_FACE_DETECT = True
YOLO_PERSON_CONF          = 0.30        # person detection confidence (lower = catch farther people)
YOLO_PERSON_PADDING       = 0.12        # expand person box before face search
FACE_DEDUP_IOU            = 0.45        # merge duplicate face boxes from full-frame + crop passes

# ---------------------------------------------------------------
# ANTI-SPOOFING / LIVENESS DETECTION
# ---------------------------------------------------------------
ENABLE_ANTI_SPOOFING  = False
LIVENESS_THRESHOLD    = 0.85            # Minimum score to be considered a real person (0.0 to 1.0)

# ---------------------------------------------------------------
# DEDUPLICATION COOLDOWNS
# ---------------------------------------------------------------
COOLDOWN_MINUTES         = 1
UNKNOWN_COOLDOWN_MINUTES = 0

# ---------------------------------------------------------------
# PERSON RE-IDENTIFICATION  (OSNet via torchreid / FastReID)
# ---------------------------------------------------------------
REID_MODEL_NAME        = "osnet_x1_0"   # model architecture
REID_SIMILARITY_THRESH = 0.55           # cosine similarity threshold (OSNet quality)
REID_GALLERY_MAX_AGE   = 14400          # 4 hours before a gallery embedding expires (keeps identity all morning)
REID_OSNET_ONLY        = False          # if True, disable histogram fallback (requires torchreid)

# ---------------------------------------------------------------
# BYTETRACK  (tracker settings)
# ---------------------------------------------------------------
BYTETRACK_TRACK_THRESH   = 0.5
BYTETRACK_TRACK_BUFFER   = 30   # frames before a track is considered lost
BYTETRACK_MATCH_THRESH   = 0.8
BYTETRACK_MIN_BOX_AREA   = 10   # px² minimum bounding box area
BYTETRACK_FRAME_RATE     = 10   # estimated camera FPS fed to ByteTrack

# ---------------------------------------------------------------
# ACTIVITY RECOGNITION
# ---------------------------------------------------------------
ENABLE_ACTIVITY_ANALYSIS      = True
ACTIVITY_CONFIDENCE_THRESHOLD = 0.45

# Idle threshold: if a person hasn't moved (IOU > threshold) for this
# many consecutive frames, mark them as idle.
IDLE_FRAME_COUNT  = 30
IDLE_IOU_THRESH   = 0.85

# ---------------------------------------------------------------
# 8-HOUR WORK DAY TARGET
# Flexible Workday Timings
WORKDAY_START_TIME = "09:00:00"
WORKDAY_END_TIME   = "21:00:00"
TARGET_WORK_HOURS  = 7.0

MONTHLY_WORKING_DAYS  = 22

# ---------------------------------------------------------------
# EMAIL ALERTS  (optional)
# ---------------------------------------------------------------
ENABLE_EMAIL_ALERTS = False
ADMIN_EMAIL         = "admin@example.com"
SMTP_SERVER         = "smtp.gmail.com"
SMTP_PORT           = 587
EMAIL_USERNAME      = "your_email@gmail.com"
EMAIL_PASSWORD      = "your_app_password"

# ---------------------------------------------------------------
# LEGACY COMPAT — activity_processor.py still imports these
# ---------------------------------------------------------------
OFFICE_CAMERA_URLS = []   # legacy office camera config (unused by new engine)

# ---------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------
import os

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
EMPLOYEE_DATA_DIR  = os.path.join(BASE_DIR, "enrollment")
DATABASE_PATH      = os.path.join(BASE_DIR, "data", "attendance.db")
LOG_FILE           = os.path.join(BASE_DIR, "logs", "system.log")
EMBEDDINGS_FILE    = os.path.join(BASE_DIR, "enrollment", "embeddings.pkl")
REID_GALLERY_FILE  = os.path.join(BASE_DIR, "enrollment", "reid_gallery.pkl")

# ---------------------------------------------------------------
# EMPLOYEE ENROLLMENT DATABASE  (SQLite — Phase 3 upgrade)
# ---------------------------------------------------------------
EMPLOYEE_DB_PATH   = os.path.join(BASE_DIR, "data", "employees.db")

# ---------------------------------------------------------------
# DATABASE URL CONFIGURATION (For SQLAlchemy Shared Databases)
# ---------------------------------------------------------------
DATABASE_URL       = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    # Fallback to local SQLite using the paths above
    ATTENDANCE_DB_URL = f"sqlite:///{DATABASE_PATH.replace(os.sep, '/')}"
    EMPLOYEE_DB_URL   = f"sqlite:///{EMPLOYEE_DB_PATH.replace(os.sep, '/')}"
else:
    # Use the shared remote database
    ATTENDANCE_DB_URL = DATABASE_URL
    EMPLOYEE_DB_URL   = DATABASE_URL

# ---------------------------------------------------------------
# EMPLOYEE GALLERY  (pre-extracted face crops + augmented images)
# Each sub-folder name becomes the employee display name.
# ---------------------------------------------------------------
GALLERY_DIR        = os.path.join(BASE_DIR, "employee gallery")
