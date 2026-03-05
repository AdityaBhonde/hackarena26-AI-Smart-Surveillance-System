# ==============================
# Backend/detection/weapon.py (Corrected: Blurring Removed)
# ==============================
import time
from datetime import datetime
import traceback
from typing import Tuple, Optional

import cv2
import numpy as np

import shared_state as state
from utils.telegram_utils import send_telegram_alert
from utils.db_utils import save_alert_to_db

# -------------------------
# Safe helpers for YOLO 'box' formats (UNTOUCHED)
# -------------------------
def _safe_get_conf_and_cls(box) -> Tuple[Optional[float], Optional[int]]:
    try:
        conf_val = None
        if hasattr(box, "conf"):
            c = box.conf
            try:
                conf_val = float(c[0].item()) if hasattr(c, "__len__") else float(c)
            except Exception:
                conf_val = float(c)

        cls_val = None
        if hasattr(box, "cls"):
            cl = box.cls
            try:
                cls_val = int(cl[0].item()) if hasattr(cl, "__len__") else int(cl)
            except Exception:
                cls_val = int(cl)

        return conf_val, cls_val
    except Exception:
        return None, None

def _safe_get_xyxy(box) -> Optional[Tuple[int, int, int, int]]:
    try:
        if hasattr(box, "xyxy"):
            pts = box.xyxy[0] if hasattr(box.xyxy, "__len__") else box.xyxy
            xy = pts.int().tolist()
            return int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
    except Exception:
        pass
    return None

# -------------------------
# Color mapping for subclasses (UNTOUCHED)
# -------------------------
DEFAULT_COLOR = (255, 255, 255)
CLASS_COLOR_MAP = {
    "gun": (0, 0, 255),
    "pistol": (0, 0, 255),
    "revolver": (0, 0, 255),
    "firearm": (0, 0, 255),
    "rifle": (0, 0, 255),
    "knife": (0, 165, 255),
    "blade": (0, 165, 255),
    "weapon": (255, 0, 0),
}

# ==================================================================================
# WEAPON DETECTION THREAD
# ==================================================================================
def weapon_detection():
    print("[weapon] Weapon detection thread started.")

    while True:
        if getattr(state, "detection_active", False) and getattr(state, "yolo_weapon_model", None) is not None:
            break
        time.sleep(0.1)

    try:
        print(f"[weapon] Loaded YOLO Weapon Model classes: {state.yolo_weapon_model.names}")
    except Exception:
        print("[weapon] WARNING: Could not print model.names")

    # ✅ Set exactly to 0.45 as requested
    MIN_CONF = 0.45 
    VALID_CLASS_KEYWORDS = ["gun", "knife", "pistol", "revolver", "firearm", "rifle", "weapon"]
    COOLDOWN = getattr(state, "ALERT_COOLDOWN", 12)

    last_alert_time = None

    while True:
        try:
            with state.frame_lock:
                frame = state.processed_frames.get("crowd")

            if frame is None:
                ok, cam_frame = state.camera_manager.read()
                if not ok or cam_frame is None:
                    time.sleep(0.01)
                    continue
                frame = cam_frame.copy()
            else:
                frame = frame.copy()

            annotated = frame.copy()

            results = state.yolo_weapon_model(frame, conf=MIN_CONF)

            if not results:
                with state.frame_lock:
                    state.processed_frames["weapon"] = annotated
                time.sleep(0.01)
                continue

            res = results[0]
            boxes = getattr(res, "boxes", None)

            weapon_detected = False
            detected_name = None
            detected_conf = None
            detected_box = None

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    conf_val, cls_val = _safe_get_conf_and_cls(box)
                    if conf_val is None:
                        continue

                    try:
                        name = state.yolo_weapon_model.names.get(cls_val, str(cls_val))
                    except Exception:
                        name = str(cls_val)

                    name = str(name).lower()

                    if any(k in name for k in VALID_CLASS_KEYWORDS) and conf_val >= MIN_CONF:
                        xy = _safe_get_xyxy(box)
                        if xy:
                            x1, y1, x2, y2 = xy
                            box_area = max(0, (x2 - x1) * (y2 - y1))
                            min_box_area = getattr(state, "WEAPON_MIN_BOX_AREA", 1500)
                            if box_area < min_box_area:
                                continue

                            weapon_detected = True
                            detected_name = name
                            detected_conf = float(conf_val)
                            detected_box = xy
                            break

            try:
                if hasattr(res, "plot"):
                    annotated = res.plot(annotated)
            except Exception:
                pass

            if detected_box is not None:
                x1, y1, x2, y2 = detected_box
                color = DEFAULT_COLOR
                for k, c in CLASS_COLOR_MAP.items():
                    if k in detected_name:
                        color = c
                        break

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_text = f"{detected_name} {detected_conf:.2f}"
                cv2.rectangle(annotated, (x1, y2 - 24), (x2, y2), color, -1)
                cv2.putText(annotated, label_text, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # =============================
            # ALERT + CLIP TRIGGER
            # =============================
            if weapon_detected and detected_name is not None:
                now = time.time()
                if not last_alert_time or (now - last_alert_time >= COOLDOWN):
                    last_alert_time = now

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    alert_text = f"🚨 WEAPON DETECTED: {detected_name.upper()} ({detected_conf:.2f}) at {timestamp}"

                    # ✅ Blurring logic removed; sending clear annotated frame
                    try:
                        send_telegram_alert(alert_text, annotated)
                    except Exception as e:
                        print(f"[weapon] Telegram send failed: {e}")

                    try:
                        save_alert_to_db(
                            alert_type="Weapon",
                            sub_type=str(detected_name),
                            confidence=float(detected_conf),
                            people_count=None,
                            person_name=None,
                            violence_detected=(getattr(state, "last_violence_info", "Safe") != "Safe"),
                        )
                    except Exception as e:
                        print(f"[weapon] DB save error: {e}")

                    state.clip_queue.put({"alert_type": "Weapon"})

                    with state.status_lock:
                        state.last_weapon_detection_time = now
                        state.last_weapon_info = f"{detected_name} ({detected_conf:.2f})"
                        state.last_weapon_confidence = detected_conf

            if last_alert_time and (time.time() - last_alert_time > COOLDOWN):
                with state.status_lock:
                    state.last_weapon_info = "Safe"
                    state.last_weapon_confidence = None

            with state.frame_lock:
                state.processed_frames["weapon"] = annotated

        except Exception as e:
            print(f"[weapon] Unexpected error: {e}")
            traceback.print_exc()

        time.sleep(0.01)
