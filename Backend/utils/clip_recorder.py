# ============================================================
# utils/clip_recorder.py (FIXED: Browser-Compatible Codec)
# ============================================================

import os
import cv2
import time
import threading
from datetime import datetime
import shared_state as state

PRE_SECONDS = 3
POST_SECONDS = 5
FPS = 20
OUTPUT_DIR = "recorded_clips"

os.makedirs(OUTPUT_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ------------------------------------------------------------
# PRIVACY BLUR FUNCTIONS
# ------------------------------------------------------------
def blur_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        roi = image[y:y + h, x:x + w]
        if roi.size > 0:
            image[y:y + h, x:x + w] = cv2.GaussianBlur(
                roi,
                (99, 99),
                30
            )
    return image


def blur_full_frame(image):
    return cv2.GaussianBlur(image, (99, 99), 30)


# ------------------------------------------------------------
# CLIP RECORDER THREAD
# ------------------------------------------------------------
def clip_recorder_worker():
    print("[clip] recorder started")

    while True:
        task = state.clip_queue.get()
        if task is None:
            break

        try:
            alert_type = task.get("alert_type", "Unknown")
            snapshot = task.get("frame_snapshot", None)
            timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

            # -------------------------
            # PRE FRAMES
            # -------------------------
            with state.frame_lock:
                pre_frames = [f.copy() for f in list(state.frame_buffer) if f is not None]

            # -------------------------
            # POST FRAMES
            # -------------------------
            post_frames = []
            for _ in range(POST_SECONDS * FPS):
                with state.frame_lock:
                    frame = (
                        state.latest_frame.copy()
                        if state.latest_frame is not None
                        else snapshot
                    )
                if frame is not None:
                    post_frames.append(frame)
                time.sleep(1 / FPS)

            all_frames = pre_frames + post_frames
            if len(all_frames) == 0:
                state.clip_queue.task_done()
                continue

            # ------------------------------------------------
            # PROCESS FRAMES
            # ------------------------------------------------
            processed_frames = []
            for frame in all_frames:
                frame = frame.copy()
                if alert_type == "Crowd":
                    frame = blur_full_frame(frame)
                else:
                    frame = blur_faces(frame)
                processed_frames.append(frame)

            # ------------------------------------------------
            # GET FRAME SIZE
            # ------------------------------------------------
            height, width = processed_frames[0].shape[:2]
            filename = f"{alert_type}--{timestamp}.mp4"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # ------------------------------------------------
            # VIDEO WRITER (FIXED FOR WEB PLAYBACK)
            # ------------------------------------------------
            # ✅ Change: Using 'avc1' (H.264) for browser compatibility
            fourcc = cv2.VideoWriter_fourcc(*"avc1")

            writer = cv2.VideoWriter(
                filepath,
                fourcc,
                FPS,
                (width, height)
            )

            # Fallback to X264 if avc1 is not available on your system
            if not writer.isOpened():
                print("[clip] avc1 failed, trying X264...")
                fourcc = cv2.VideoWriter_fourcc(*"X264")
                writer = cv2.VideoWriter(filepath, fourcc, FPS, (width, height))

            if not writer.isOpened():
                print("[clip] ERROR: Both avc1 and X264 failed. falling back to mp4v")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(filepath, fourcc, FPS, (width, height))

            time.sleep(0.1)

            for frame in processed_frames:
                if frame is None:
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)

            writer.release()
            print(f"[clip] saved browser-compatible video: {filename}")

        except Exception as e:
            print("[clip] error:", e)

        state.clip_queue.task_done()


# ------------------------------------------------------------
# START THREAD
# ------------------------------------------------------------
def start_clip_recorder():
    if not any(t.name == "ClipRecorderThread" for t in threading.enumerate()):
        t = threading.Thread(
            target=clip_recorder_worker,
            daemon=True,
            name="ClipRecorderThread"
        )
        t.start()
