# ==============================
# detection/crowd.py (Precise Face Blur + Original Logic)
# ==============================

import time
import cv2
import threading
from deep_sort_realtime.deepsort_tracker import DeepSort
import shared_state as state
from utils.telegram_utils import send_telegram_alert
from utils.db_utils import save_alert_to_db

# -----------------------------
# FACE DETECTOR (Privacy Blur)
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -----------------------------
# DeepSORT Tracker
# -----------------------------
tracker = DeepSort(
    max_age=100,
    n_init=1,
    max_cosine_distance=0.4,
    nn_budget=200
)

# -----------------------------
# PRECISE FACE BLUR FUNCTION
# -----------------------------
def blur_all_faces(image):
    """
    Highly precise face blurring. 
    Adjusted scaleFactor and minNeighbors to detect more faces in a crowd.
    """
    if image is None:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Improved parameters for better detection in crowded scenes
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,  # Smaller steps for higher precision
        minNeighbors=4,     # Lowered slightly to pick up more faces in background
        minSize=(20, 20)    # Detect smaller faces (farther away)
    )

    for (x, y, w, h) in faces:
        # Ensure coordinates are within image boundaries
        x, y = max(0, x), max(0, y)
        
        # Extract the exact face ROI
        roi = image[y:y+h, x:x+w]
        
        if roi.size > 0:
            # Apply a heavy blur only to the face area
            # (99, 99) ensures the face is unrecognizable
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
            image[y:y+h, x:x+w] = blurred_roi

    return image

# -----------------------------
# ALERT WORKER THREAD
# -----------------------------
def alert_worker():
    while True:
        task = state.alert_queue.get()
        if task is None:
            break

        alert_type, msg, frame, extra = task

        try:
            sending_frame = frame.copy()

            # ✅ Precise Blur only for Crowd Overload on Telegram
            if alert_type == "Crowd":
                sending_frame = blur_all_faces(sending_frame)

            send_telegram_alert(msg, sending_frame)

            # Database logging (Untouched)
            if alert_type == "Loitering":
                save_alert_to_db(alert_type="Loitering", person_id=extra)
            elif alert_type == "Crowd":
                save_alert_to_db(alert_type="Crowd", people_count=extra)

        except Exception as e:
            print(f"[Alert Worker Error] {e}")

        state.alert_queue.task_done()

# Start the worker thread
if not any(t.name == "CrowdAlertWorker" for t in threading.enumerate()):
    threading.Thread(target=alert_worker, daemon=True, name="CrowdAlertWorker").start()

# -----------------------------
# MAIN DETECTION FUNCTION
# -----------------------------
def crowd_detection():
    print("[INFO] Crowd + Loitering module started.")

    last_loiter_alert = 0
    last_crowd_alert = 0
    frame_skip_counter = 0

    while state.detection_active and state.yolo_crowd_model:

        with state.frame_lock:
            frame = state.latest_frame.copy() if state.latest_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        frame_skip_counter += 1
        if frame_skip_counter % 2 != 0:
            continue

        annotated = frame.copy()

        # 1. PERSON DETECTION
        results = state.yolo_crowd_model(frame, conf=0.30, verbose=False)
        detections = []

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1, y1, w, h], float(box.conf[0]), "person"))

        # 2. TRACKING
        tracks = tracker.update_tracks(detections, frame=frame)
        current_time = time.time()
        active_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            active_ids.add(tid)

            if tid not in state.person_entry_times:
                state.person_entry_times[tid] = current_time

            duration = current_time - state.person_entry_times[tid]
            l, t, r, b = map(int, track.to_ltrb())

            # -----------------------------
            # LOITERING CHECK (Untouched)
            # -----------------------------
            if duration >= state.LOITER_THRESHOLD:
                cv2.rectangle(annotated, (l, t), (r, b), (0, 0, 255), 2)
                cv2.putText(annotated, f"LOITERING {int(duration)}s", (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if current_time - last_loiter_alert >= state.ALERT_COOLDOWN:
                    msg = f"🚨 LOITERING ALERT\nPerson ID: {tid}\nDuration: {int(duration)}s"
                    state.alert_queue.put(("Loitering", msg, frame.copy(), tid))
                    state.clip_queue.put({"alert_type": "Loitering"})
                    last_loiter_alert = current_time

        # -----------------------------
        # CROWD OVERLOAD CHECK
        # -----------------------------
        people_count = len(active_ids)
        state.crowd_count = str(people_count)

        if people_count > 35 and (current_time - last_crowd_alert >= state.ALERT_COOLDOWN):
            msg = f"🚨 CROWD OVERLOAD ALERT\nPeople Count: {people_count}"
            state.alert_queue.put(("Crowd", msg, frame.copy(), people_count))
            state.clip_queue.put({"alert_type": "Crowd"})
            last_crowd_alert = current_time

        state.person_entry_times = {k: v for k, v in state.person_entry_times.items() if k in active_ids}

        # UI for Live Feed
        cv2.putText(annotated, f"Count: {people_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        with state.frame_lock:
            state.processed_frames['crowd'] = annotated

        time.sleep(0.01)

