# ============================================================
# shared_state.py
# ============================================================
import threading
import queue
from collections import deque   # ✅ NEW (for frame buffer)

# ---------------- THREAD LOCKS ----------------
# Use frame_lock when updating or reading latest_frame or processed_frames
frame_lock = threading.Lock()

# Use status_lock if you are updating global status variables across threads
status_lock = threading.Lock()

# ---------------- MASTER FRAME ----------------
# This holds the raw frame from the camera to be shared by all modules
latest_frame = None

# ---------------- FRAME BUFFER (NEW) ----------------
# Stores last ~3 seconds of frames (assuming ~30 FPS → 90 frames)
frame_buffer = deque(maxlen=90)

# ---------------- DETECTION STATUS ----------------
# Holds the resulting annotated frames for the Flask web feed
processed_frames = {
    'crowd': None,
    'weapon': None,
    'violence': None,
    'criminal': None
}

detection_active = False

# ---------------- MODELS & MANAGERS ----------------
yolo_crowd_model = None
yolo_weapon_model = None
violence_model = None
camera_manager = None

# ---------------- CROWD DATA ----------------
crowd_count = "0"
crowd_history = []

# ---------------- LOITERING DATA ----------------
# person_entry_times tracks when a specific ID was first seen
person_entry_times = {}
LOITER_THRESHOLD = 20  # Seconds before an alert is triggered

# ---------------- ALERT SYSTEM (CRITICAL FOR LAG FIX) ----------------
# alert_queue allows us to send Telegram/DB alerts in the background
alert_queue = queue.Queue()

# ---------------- CLIP RECORDING SYSTEM (NEW) ----------------
# Used by clip_recorder.py to trigger recording
clip_queue = queue.Queue()

# ---------------- ALERT COOLDOWN STATES ----------------
# These prevent the system from spamming your phone
last_weapon_detection_time = 0
last_weapon_info = "Safe"
last_violence_detection_time = 0
last_violence_info = "Safe"

# ---------------- CONSTANTS ----------------
ALERT_COOLDOWN = 12  # Minimum seconds between two alerts
DETECTION_CONF_THRESHOLD = 0.20