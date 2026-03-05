# ==============================
# Backend/detection/criminal.py
# ==============================

import time
import cv2
import face_recognition
import pickle
from datetime import datetime
import shared_state as state
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

        # Get frame from previous pipeline (weapon detection output)
        with state.frame_lock:
            frame = state.processed_frames.get("weapon")

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

        # Run detection every N frames
        if frame_count % PROCESS_EVERY_N_FRAMES == 0 and len(known_encodings) > 0:

            # Resize for speed
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            telegram_frame = frame.copy()

            criminal_found = False
            best_match_data = None
            best_box = None

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                distances = face_recognition.face_distance(known_encodings, face_encoding)

                name = "Unknown"
                dist_val = 1.0

                if len(distances) > 0:

                    best_idx = int(distances.argmin())

                    if distances[best_idx] <= TOLERANCE:
                        name = known_names[best_idx]
                        dist_val = distances[best_idx]

                # scale box back
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if name != "Unknown":

                    criminal_found = True

                    best_match_data = {
                        "name": name,
                        "dist": dist_val
                    }

                    best_box = (top, right, bottom, left)

                    # draw box on live feed
                    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 0, 255), 2)

                    cv2.putText(
                        annotated,
                        f"SUSPECT: {name}",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                    # draw on telegram frame
                    cv2.rectangle(telegram_frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    cv2.putText(
                        telegram_frame,
                        f"SUSPECT: {name}",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                else:

                    # blur unknown faces for privacy
                    y1 = max(0, top)
                    y2 = min(frame.shape[0], bottom)

                    x1 = max(0, left)
                    x2 = min(frame.shape[1], right)

                    roi = telegram_frame[y1:y2, x1:x2]

                    if roi.size > 0:
                        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                        telegram_frame[y1:y2, x1:x2] = blurred

            # =============================
            # ALERT TRIGGER
            # =============================
            if criminal_found:

                now = time.time()

                if now - last_alert_time >= state.ALERT_COOLDOWN:

                    name = best_match_data["name"]
                    conf = 1.0 - best_match_data["dist"]

                    # update dashboard state
                    with state.status_lock:
                        state.last_violence_detection_time = now
                        state.last_violence_info = f"CRIMINAL: {name}"

                    # send alert image
                    state.alert_queue.put(("Criminal", name, telegram_frame, conf))

                    # 🔥 FIXED CLIP SYNC
                    state.clip_queue.put(
                        {
                            "alert_type": "Criminal",
                            "frame_snapshot": frame.copy(),
                            "exclude_box": best_box,
                        }
                    )

                    try:
                        save_alert_to_db(
                            alert_type="Criminal",
                            sub_type=name,
                            person_name=name,
                            confidence=round(conf, 2),
                        )
                    except:
                        pass

                    last_alert_time = now

        # update live feed frame
        with state.frame_lock:
            state.processed_frames["violence"] = annotated

        time.sleep(0.01)
