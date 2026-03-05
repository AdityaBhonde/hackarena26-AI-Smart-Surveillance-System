# ============================================================
# utils/clip_recorder.py
# ============================================================

import os
import cv2
import time
import threading
from datetime import datetime
from collections import deque
import shared_state as state

# -----------------------------
# CONFIGURATION
# -----------------------------
PRE_SECONDS = 3
POST_SECONDS = 5
FPS = 30
OUTPUT_DIR = "recorded_clips"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FACE DETECTOR (For Privacy)
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def blur_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        roi = image[y:y+h, x:x+w]
        if roi.size > 0:
            blurred = cv2.GaussianBlur(roi, (51, 51), 0)
            image[y:y+h, x:x+w] = blurred
    return image

def blur_full_frame(image):
    return cv2.GaussianBlur(image, (51, 51), 0)

# ============================================================
# CLIP RECORDER WORKER
# ============================================================
def clip_recorder_worker():
    print("[clip] Clip recorder thread started.")

    while True:
        task = state.clip_queue.get()

        if task is None:
            break

        try:
            alert_type = task.get("alert_type", "Unknown")

            print(f"[clip] Recording clip for {alert_type}")

            # -----------------------------
            # STEP 1: Get PRE-ALERT frames
            # -----------------------------
            with state.frame_lock:
                pre_frames = list(state.frame_buffer)

            # -----------------------------
            # STEP 2: Collect POST frames
            # -----------------------------
            post_frames = []
            frames_needed = POST_SECONDS * FPS
            collected = 0

            while collected < frames_needed:
                with state.frame_lock:
                    frame = state.latest_frame.copy() if state.latest_frame is not None else None

                if frame is not None:
                    post_frames.append(frame)
                    collected += 1

                time.sleep(1 / FPS)

            # -----------------------------
            # STEP 3: Combine frames
            # -----------------------------
            all_frames = pre_frames + post_frames

            if len(all_frames) == 0:
                print("[clip] No frames available, skipping.")
                continue

            height, width, _ = all_frames[0].shape

            # -----------------------------
            # 🔐 PRIVACY ENFORCEMENT
            # -----------------------------
            processed_frames = []

            for frame in all_frames:

                frame_copy = frame.copy()

                if alert_type == "Crowd":
                    # Full frame blur
                    frame_copy = blur_full_frame(frame_copy)

                elif alert_type == "Loitering":
                    frame_copy = blur_faces(frame_copy)

                elif alert_type == "Weapon":
                    frame_copy = blur_faces(frame_copy)

                elif alert_type == "Criminal":
                    # Criminal selective blur handled in detection frame,
                    # so blur faces here as safety fallback
                    frame_copy = blur_faces(frame_copy)

                processed_frames.append(frame_copy)

            # -----------------------------
            # STEP 4: Create filename
            # -----------------------------
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{alert_type}_{timestamp}.mp4"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # -----------------------------
            # STEP 5: Write video
            # -----------------------------
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, FPS, (width, height))

            for frame in processed_frames:
                out.write(frame)

            out.release()

            print(f"[clip] Saved: {filepath}")

        except Exception as e:
            print(f"[clip] Error: {e}")

        state.clip_queue.task_done()

# ============================================================
# START THREAD
# ============================================================
def start_clip_recorder():
    threading.Thread(
        target=clip_recorder_worker,
        daemon=True
    ).start()