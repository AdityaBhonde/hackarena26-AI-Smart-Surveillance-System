# ==============================
# Backend/detection/criminal.py (Fixed Dashboard + Aggregator)
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
    last_alert_time = 0 # Initialize to 0 for numerical comparison

    while state.detection_active:
        with state.frame_lock:
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
        annotated = frame.copy()
        
        if frame_count % PROCESS_EVERY_N_FRAMES == 0 and len(known_encodings) > 0:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            face_data = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) == 0: continue

                best_idx = int(distances.argmin())
                match = distances[best_idx] <= TOLERANCE
                name = known_names[best_idx] if match else "Unknown"

                # Scale back to original size
                top, right, bottom, left = top*4, right*4, bottom*4, left*4

                if name != "Unknown":
                    # ✅ DRAW RED BOX ON LIVE FEED (Annotated)
                    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(annotated, f"SUSPECT: {name}", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Store data for reporting
                    face_data.append({"name": name, "distance": distances[best_idx], "frame": frame.copy()})

            # ===============================
            # ✅ CORRECTED ALERT + DASHBOARD LOGIC
            # ===============================
            for face in face_data:
                name = face["name"]
                now = time.time()
                
                if now - last_alert_time >= state.ALERT_COOLDOWN:
                    # 1. Update Dashboard Status
                    with state.status_lock:
                        state.last_violence_detection_time = now
                        state.last_violence_info = f"CRIMINAL: {name}"

                    # 2. Report to Aggregator (This handles Telegram and Blurring)
                    state.alert_queue.put(("Criminal", name, face["frame"], 1.0 - face["distance"]))

                    # 3. Trigger Clip Recording
                    state.clip_queue.put({"alert_type": "Criminal"})

                    # 4. Save to DB
                    try:
                        save_alert_to_db(
                            alert_type="Criminal",
                            sub_type=name,
                            person_name=name,
                            confidence=round(1.0 - face["distance"], 2)
                        )
                    except: pass

                    last_alert_time = now
                    break # Alert only once per cycle

        with state.frame_lock:
            state.processed_frames['violence'] = annotated

        time.sleep(0.01)
