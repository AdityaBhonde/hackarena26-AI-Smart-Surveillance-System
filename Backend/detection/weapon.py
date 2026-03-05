# ==============================
# Backend/detection/weapon.py (Strict Confidence + Web Alert Fix)
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

def _safe_get_conf_and_cls(box) -> Tuple[Optional[float], Optional[int]]:
    try:
        conf_val = None
        if hasattr(box, "conf"):
            c = box.conf
            conf_val = float(c[0].item()) if hasattr(c, "__len__") else float(c)
        cls_val = None
        if hasattr(box, "cls"):
            cl = box.cls
            cls_val = int(cl[0].item()) if hasattr(cl, "__len__") else int(cl)
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

CLASS_COLOR_MAP = {
    "gun": (0, 0, 255), "pistol": (0, 0, 255), "revolver": (0, 0, 255),
    "firearm": (0, 0, 255), "rifle": (0, 0, 255), "knife": (0, 165, 255),
    "blade": (0, 165, 255), "weapon": (255, 0, 0),
}

def weapon_detection():
    print("[weapon] Weapon detection thread started.")

    while True:
        if getattr(state, "detection_active", False) and getattr(state, "yolo_weapon_model", None) is not None:
            break
        time.sleep(0.1)

    # ✅ STRICT THRESHOLD: Ignore everything below 0.75
    MIN_CONF = 0.75 
    VALID_CLASS_KEYWORDS = ["gun", "knife", "pistol", "revolver", "firearm", "rifle", "weapon"]
    COOLDOWN = getattr(state, "ALERT_COOLDOWN", 12)

    last_alert_time = 0

    while True:
        try:
            with state.frame_lock:
                frame = state.processed_frames.get("crowd")
            
            if frame is None:
                ok, cam_frame = state.camera_manager.read()
                if not ok: continue
                frame = cam_frame.copy()
            else:
                frame = frame.copy()

            annotated = frame.copy()

            # Pass the strict threshold directly to the YOLO model
            results = state.yolo_weapon_model(frame, conf=MIN_CONF, verbose=False)

            weapon_detected = False
            best_hit = None

            if results and results[0].boxes:
                for box in results[0].boxes:
                    conf_val, cls_val = _safe_get_conf_and_cls(box)
                    
                    if conf_val is not None and conf_val >= MIN_CONF:
                        name = state.yolo_weapon_model.names.get(cls_val, str(cls_val)).lower()
                        
                        if any(k in name for k in VALID_CLASS_KEYWORDS):
                            xy = _safe_get_xyxy(box)
                            if xy:
                                x1, y1, x2, y2 = xy
                                if (x2 - x1) * (y2 - y1) > 1500: 
                                    weapon_detected = True
                                    best_hit = {"name": name, "conf": conf_val, "box": xy}
                                    
                                    # ✅ ONLY DRAW IF CONFIDENCE IS ABOVE 0.75
                                    color = CLASS_COLOR_MAP.get(name, (255, 255, 255))
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                                    label = f"{name.upper()} {conf_val:.2f}"
                                    cv2.putText(annotated, label, (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                    break 

            # =============================
            # AGGREGATED ALERT TRIGGER
            # =============================
            if weapon_detected and best_hit:
                now = time.time()
                if now - last_alert_time >= COOLDOWN:
                    # ✅ WEB ALERT FIX: Update dashboard timestamp and info
                    with state.status_lock:
                        state.last_weapon_detection_time = now  # This triggers the webpage alert
                        state.last_weapon_info = f"{best_hit['name']} ({best_hit['conf']:.2f})"
                        state.last_weapon_confidence = best_hit['conf']
                    
                    # Report to the Aggregator Queue
                    state.alert_queue.put(("Weapon", f"{best_hit['name'].upper()} ({best_hit['conf']:.2f})", frame.copy(), best_hit['conf']))
                    state.clip_queue.put({"alert_type": "Weapon"})
                    
                    last_alert_time = now

            elif time.time() - last_alert_time > COOLDOWN:
                with state.status_lock:
                    state.last_weapon_info = "Safe"

            with state.frame_lock:
                state.processed_frames["weapon"] = annotated

        except Exception as e:
            print(f"[weapon] Error: {e}")
            traceback.print_exc()

        time.sleep(0.01)
