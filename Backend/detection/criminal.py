# ==============================
# Backend/detection/criminal.py (PRECISE BLUR FIX + AGGREGATOR SYNC)
# ==============================
import time
import cv2
import face_recognition
import pickle
from datetime import datetime
import shared_state as state
from utils.telegram_utils import send_telegram_alert
from utils.db_utils import save_alert_to_db
from pathlib import Path

# Configuration
PROCESS_EVERY_N_FRAMES = 2
TOLERANCE = 0.45

# Load known encodings
BASE_DIR = Path(__file__).resolve().parent.parent
ENCODINGS_PATH = BASE_DIR / "models" / "face_recognition" / "encodings.pkl"

known_encodings = []
known_names = []
try:
    with open(str(ENCODINGS_PATH), "rb") as f:
        data = pickle.load(f)
        known_encodings = data.get("encodings", [])
        known_names = data.get("names", [])
    print(f"[criminal] Loaded {len(known_names)} known identities.")
except Exception as e:
    print(f"[criminal] Warning: {e}")

def criminal_detection():
    print("[INFO] Criminal detection thread started.")
    frame_count = 0
    last_alert_time = 0 

    while state.detection_active:
        with state.frame_lock:
            # Maintain the chain from the weapon process
            frame = state.processed_frames.get('weapon')

        if frame is None:
            ok, tmp = state.camera_manager.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = tmp.copy()
        else:
            frame = frame.copy()

        frame_count += 1
        annotated = frame.copy() # Clear frame for Live Feed
        
        if frame_count % PROCESS_EVERY_N_FRAMES == 0 and len(known_encodings) > 0:
            # Step 1: Process at 1/4 size for speed
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Step 2: Get locations from face_recognition
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            # This frame is specifically for the Telegram alert
            telegram_frame = frame.copy()
            criminal_found_in_frame = False
            best_match_data = None
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                name = "Unknown"
                dist_val = 1.0
                if len(distances) > 0:
                    best_idx = int(distances.argmin())
                    if distances[best_idx] <= TOLERANCE:
                        name = known_names[best_idx]
                        dist_val = distances[best_idx]

                # Scale coordinates back to original size (x4)
                top, right, bottom, left = top*4, right*4, bottom*4, left*4

                if name != "Unknown":
                    criminal_found_in_frame = True
                    # 1. Draw RED BOX on the Clear Live Feed
                    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(annotated, f"SUSPECT: {name}", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # 2. Draw on the Telegram frame (Keep criminal face clear)
                    cv2.rectangle(telegram_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(telegram_frame, f"SUSPECT: {name}", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    best_match_data = {"name": name, "dist": dist_val}
                
                else:
                    # 3. BLUR UNKNOWN FACES (Only on the Telegram version)
                    y1, y2 = max(0, top), min(frame.shape[0], bottom)
                    x1, x2 = max(0, left), min(frame.shape[1], right)
                    
                    face_roi = telegram_frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        # Heavy Gaussian Blur using sync'd coordinates
                        blurred_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                        telegram_frame[y1:y2, x1:x2] = blurred_roi

            # ===============================
            # ALERT TRIGGER (AGGREGATOR SYNC)
            # ===============================
            if criminal_found_in_frame:
                now = time.time()
                if now - last_alert_time >= state.ALERT_COOLDOWN:
                    name = best_match_data["name"]
                    conf = 1.0 - best_match_data["dist"]

                    # Update Dashboard Status
                    with state.status_lock:
                        state.last_violence_detection_time = now
                        state.last_violence_info = f"CRIMINAL: {name}"

                    # ✅ THIS IS THE FIX: Send 'telegram_frame' to the alert_queue
                    # The worker in crowd.py will send this exact blurred image
                    state.alert_queue.put(("Criminal", name, telegram_frame, conf))

                    state.clip_queue.put({"alert_type": "Criminal"})

                    try:
                        save_alert_to_db(alert_type="Criminal", sub_type=name, 
                                         person_name=name, confidence=round(conf, 2))
                    except: pass

                    last_alert_time = now

        # Update the Live Feed with the unblurred but annotated frame
        with state.frame_lock:
            state.processed_frames['violence'] = annotated

        time.sleep(0.01)
