"""
ROAD SAFETY MONITOR - With Car Color Detection (5-Frame Burst on Violation)
Saves: 5 frames (2 before + current + 2 after) per violation
       + Car Color + Plate Number + Violation Type + Timestamp
"""

import cv2
import time
import torch
import threading
import os
import json
import uuid
import numpy as np
import urllib.request
import ssl
import socket
import concurrent.futures
from collections import deque
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory
from ultralytics import YOLO

# Try to import sklearn for color detection
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[WARN] scikit-learn not installed. Color detection will be limited.")
    print("[INFO] Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────
# FLASK SETUP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

METADATA_FILE = os.path.join(SNAPSHOT_DIR, "violations_metadata.json")

# ─────────────────────────────────────────────
# LOAD YOLOv8
# ─────────────────────────────────────────────
print("[INFO] Loading YOLOv8...")

MODEL_NAME = "yolov8n.pt"

try:
    model = YOLO(MODEL_NAME)
    print(f"[INFO] YOLOv8 loaded: {MODEL_NAME}")
except Exception as e:
    print(f"[ERROR] Failed to load YOLOv8: {e}")
    exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    model.to(DEVICE)
    print("[INFO] YOLOv8 running on GPU")
else:
    print("[INFO] YOLOv8 running on CPU")

print(f"[INFO] YOLOv8 ready on {DEVICE.upper()}")

# ─────────────────────────────────────────────
# LOAD EasyOCR
# ─────────────────────────────────────────────
try:
    import easyocr
    print("[INFO] Loading EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"), verbose=False)
    print("[INFO] EasyOCR ready")
    OCR_AVAILABLE = True
except ImportError:
    print("[WARN] EasyOCR not installed.")
    OCR_AVAILABLE = False
    reader = None

# ─────────────────────────────────────────────
# ROLLING FRAME BUFFER (per camera)
# ─────────────────────────────────────────────
BURST_BEFORE = 2          # frames to keep before violation
BURST_AFTER  = 2          # frames to capture after violation
BURST_TOTAL  = BURST_BEFORE + 1 + BURST_AFTER  # = 5

class FrameBuffer:
    """
    Thread-safe rolling buffer that stores the last N raw frames.
    Used to retrieve frames just before a violation was detected.
    """
    def __init__(self, maxlen=BURST_BEFORE + 1):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray):
        with self._lock:
            self._buf.append(frame.copy())

    def get_recent(self, n: int) -> list:
        """Return up to n most-recent frames (oldest first)."""
        with self._lock:
            frames = list(self._buf)
        return frames[-n:] if len(frames) >= n else frames


# One buffer per camera
frame_buffers = {
    "CAM-1": FrameBuffer(maxlen=BURST_BEFORE + 1),
    "CAM-2": FrameBuffer(maxlen=BURST_BEFORE + 1),
}

# ─────────────────────────────────────────────
# PENDING BURST CAPTURES
# ─────────────────────────────────────────────
# When a violation fires we immediately save the buffered frames (before + current),
# then schedule the "after" frames to be saved as they arrive.
#
# pending_bursts[cam_id] is a list of active BurstCapture objects.

class BurstCapture:
    """Manages saving of all 5 frames for a single violation event."""

    def __init__(self, cam_id, plate, violation, confidence, vehicle_type,
                 vehicle_id, car_color, x1, y1, x2, y2, pre_frames):
        self.cam_id       = cam_id
        self.plate        = plate
        self.violation    = violation
        self.confidence   = confidence
        self.vehicle_type = vehicle_type
        self.vehicle_id   = vehicle_id
        self.car_color    = car_color
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

        # Build unique folder for this burst
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        uid     = uuid.uuid4().hex[:6]
        safe_p  = ''.join(c for c in plate if c.isalnum()) if plate != "UNKNOWN" else "UNKNOWN"
        safe_v  = violation.replace(" ", "_")
        self.folder_name = f"{cam_id}_{safe_p}_{car_color}_{safe_v}_{ts}_{uid}"
        self.folder_path = os.path.join(SNAPSHOT_DIR, self.folder_name)
        os.makedirs(self.folder_path, exist_ok=True)

        self.saved_count  = 0          # how many frames saved so far
        self.after_needed = BURST_AFTER  # how many "after" frames still needed
        self.filenames    = []

        # Save pre-violation + current frames immediately
        for frame in pre_frames:
            self._save_frame(frame)

    def feed_after_frame(self, frame: np.ndarray) -> bool:
        """
        Call with each new raw frame after the violation.
        Returns True when all 5 frames have been saved.
        """
        if self.after_needed <= 0:
            return True
        self._save_frame(frame)
        self.after_needed -= 1
        return self.after_needed == 0

    def _save_frame(self, raw_frame: np.ndarray):
        """Crop, annotate, and save a single frame."""
        h, w = raw_frame.shape[:2]

        # Crop with margin
        margin  = SNAPSHOT_MARGIN
        cx1     = max(0, self.x1 - margin)
        cy1     = max(0, self.y1 - margin)
        cx2     = min(w, self.x2 + margin)
        cy2     = min(h, self.y2 + margin)
        roi     = raw_frame[cy1:cy2, cx1:cx2]

        if roi.size == 0:
            roi = raw_frame.copy()

        # Up-scale if too small
        rh, rw = roi.shape[:2]
        if rw < SNAPSHOT_MIN_WIDTH or rh < SNAPSHOT_MIN_HEIGHT:
            scale_w = max(SNAPSHOT_MIN_WIDTH / max(rw, 1), 1.2)
            scale_h = max(SNAPSHOT_MIN_HEIGHT / max(rh, 1), 1.2)
            roi = cv2.resize(roi, (int(rw * scale_w), int(rh * scale_h)),
                             interpolation=cv2.INTER_CUBIC)

        roi = self._annotate(roi)

        frame_num = self.saved_count + 1
        fname     = f"frame_{frame_num}.jpg"
        fpath     = os.path.join(self.folder_path, fname)
        cv2.imwrite(fpath, roi)
        self.filenames.append(fname)
        self.saved_count += 1

    def _annotate(self, img: np.ndarray) -> np.ndarray:
        """Draw information overlay on a snapshot image."""
        color_map_for_bar = {
            "RED": (0, 0, 255), "BLUE": (255, 0, 0), "GREEN": (0, 255, 0),
            "YELLOW": (0, 255, 255), "BLACK": (0, 0, 0), "WHITE": (255, 255, 255),
            "GRAY": (128, 128, 128), "SILVER": (192, 192, 192),
            "ORANGE": (0, 165, 255), "PURPLE": (128, 0, 128),
            "PINK": (203, 192, 255), "BROWN": (42, 42, 165),
            "CYAN": (255, 255, 0), "OTHER": (100, 100, 100)
        }

        bar_color = color_map_for_bar.get(self.car_color, (100, 100, 100))
        h_img, w_img = img.shape[:2]

        # Dark header bar
        bar_h   = 130
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w_img, bar_h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        # Color strip on left edge
        cv2.rectangle(img, (0, 0), (15, h_img), bar_color, -1)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        fscl  = 0.7
        thick = 2

        cv2.putText(img, f"COLOR: {self.car_color}",          (25,  28), font, fscl, bar_color,        thick)
        cv2.putText(img, f"PLATE: {self.plate}",               (25,  53), font, fscl, (0, 255, 0),      thick)
        cv2.putText(img, f"TYPE:  {self.vehicle_type.upper()}",(25,  78), font, fscl, (255, 255, 255),  thick)
        cv2.putText(img, f"VIOLATION: {self.violation.replace('_',' ')}", (25, 103), font, 0.6, (0, 165, 255), 2)

        # Frame counter
        frame_label = f"FRAME {self.saved_count + 1}/{BURST_TOTAL}"
        cv2.putText(img, frame_label, (25, 128), font, 0.55, (200, 200, 0), 2)

        # Timestamp bottom-right
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (tw, _), _ = cv2.getTextSize(ts_str, font, 0.5, 1)
        cv2.putText(img, ts_str, (w_img - tw - 10, h_img - 10), font, 0.5, (200, 200, 200), 1)

        # Border
        cv2.rectangle(img, (0, 0), (w_img - 1, h_img - 1), (0, 255, 255), 3)
        return img

    def metadata(self) -> dict:
        return {
            "folder":       self.folder_name,
            "timestamp":    datetime.now().isoformat(),
            "camera":       self.cam_id,
            "plate":        self.plate,
            "color":        self.car_color,
            "vehicle_type": self.vehicle_type,
            "violation":    self.violation.replace("_", " "),
            "confidence":   self.confidence,
            "folder_path":  self.folder_path,
            "frames":       self.filenames,
            "total_frames": BURST_TOTAL,
        }


# Active bursts per camera  { cam_id: [BurstCapture, ...] }
pending_bursts: dict[str, list] = {"CAM-1": [], "CAM-2": []}
pending_bursts_lock = threading.Lock()


# ─────────────────────────────────────────────
# CAR COLOR DETECTION (with caching)
# ─────────────────────────────────────────────
color_cache = {}
COLOR_CACHE_TTL = 3.0  # seconds


def detect_car_color(vehicle_image):
    if vehicle_image is None or vehicle_image.size == 0:
        return "UNKNOWN"

    color_ranges = {
        "RED":    [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
        "ORANGE": [(11, 50, 50), (22, 255, 255)],
        "YELLOW": [(23, 50, 50), (35, 255, 255)],
        "GREEN":  [(36, 50, 50), (85, 255, 255)],
        "BLUE":   [(86, 50, 50), (125, 255, 255)],
        "PURPLE": [(126, 50, 50), (145, 255, 255)],
        "PINK":   [(146, 50, 50), (165, 255, 255)],
        "WHITE":  [(0, 0, 200), (180, 30, 255)],
        "BLACK":  [(0, 0, 0), (180, 255, 50)],
        "GRAY":   [(0, 0, 50), (180, 50, 200)],
        "SILVER": [(0, 0, 100), (180, 30, 200)],
        "BROWN":  [(10, 50, 50), (20, 255, 150)],
        "CYAN":   [(85, 50, 50), (95, 255, 255)]
    }

    h, w = vehicle_image.shape[:2]
    if h > 200 or w > 200:
        vehicle_image = cv2.resize(vehicle_image, (200, 200))

    hsv = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2HSV)
    best_match, best_score = "UNKNOWN", 0

    for color_name, ranges in color_ranges.items():
        if len(ranges) == 4:
            l1, u1, l2, u2 = np.array(ranges[0]), np.array(ranges[1]), \
                              np.array(ranges[2]), np.array(ranges[3])
            mask = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
        else:
            mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))

        score = np.sum(mask > 0) / (hsv.shape[0] * hsv.shape[1])
        if score > best_score:
            best_score = score
            best_match = color_name

    if best_score < 0.15 and SKLEARN_AVAILABLE:
        try:
            pixels = vehicle_image.reshape(-1, 3)
            if len(pixels) > 5000:
                idx    = np.random.choice(len(pixels), 5000, replace=False)
                pixels = pixels[idx]
            kmeans       = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dom_bgr      = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
            dom_hsv      = cv2.cvtColor(np.uint8([[dom_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            for color_name, ranges in color_ranges.items():
                if len(ranges) == 4:
                    if (ranges[0][0] <= dom_hsv[0] <= ranges[1][0] or
                            ranges[2][0] <= dom_hsv[0] <= ranges[3][0]):
                        best_match = color_name
                        break
                else:
                    if ranges[0][0] <= dom_hsv[0] <= ranges[1][0]:
                        best_match = color_name
                        break
        except Exception:
            pass

    if best_match == "UNKNOWN" or best_score < 0.1:
        r, g, b = np.mean(vehicle_image, axis=(0, 1))[[2, 1, 0]]
        if   r > 200 and g > 200 and b > 200:    best_match = "WHITE"
        elif r < 50  and g < 50  and b < 50:     best_match = "BLACK"
        elif r > 200 and g < 100 and b < 100:    best_match = "RED"
        elif r < 100 and g > 200 and b < 100:    best_match = "GREEN"
        elif r < 100 and g < 100 and b > 200:    best_match = "BLUE"
        elif r > 150 and g > 150 and b < 100:    best_match = "YELLOW"
        elif abs(r-g) < 30 and abs(g-b) < 30 and r > 100: best_match = "SILVER"
        else:                                    best_match = "OTHER"

    return best_match


def get_cached_color(vehicle_id, vehicle_image):
    now = time.time()
    if vehicle_id in color_cache:
        cached_color, expiry = color_cache[vehicle_id]
        if now < expiry:
            return cached_color
    color = detect_car_color(vehicle_image)
    color_cache[vehicle_id] = (color, now + COLOR_CACHE_TTL)
    return color


# ─────────────────────────────────────────────
# COCO CLASS IDs
# ─────────────────────────────────────────────
CLS_PERSON        = 0
CLS_BICYCLE       = 1
CLS_CAR           = 2
CLS_MOTORCYCLE    = 3
CLS_BUS           = 5
CLS_TRUCK         = 7
CLS_TRAFFIC_LIGHT = 9
CLS_CELL_PHONE    = 67

VEHICLE_CLASSES  = {CLS_CAR, CLS_MOTORCYCLE, CLS_BUS, CLS_TRUCK, CLS_BICYCLE}
ALL_USED_CLASSES = VEHICLE_CLASSES | {CLS_PERSON, CLS_TRAFFIC_LIGHT, CLS_CELL_PHONE}
SNAPSHOT_CLASSES = {CLS_CAR, CLS_TRUCK, CLS_BUS, CLS_MOTORCYCLE}

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
WRONG_LANE_THRESHOLD    = 0.38
PARKING_FRAME_THRESHOLD = 150
PARKING_MIN_DURATION    = 5.0
SPEED_GROWTH_THRESHOLD  = 0.18
COOLDOWN_SECONDS        = 10
OCR_MIN_HEIGHT          = 30
OCR_EVERY_N_FRAMES      = 10
CONFIDENCE_THRESHOLD    = 0.45
CROSSED_MEMORY_SECONDS  = 15

SNAPSHOT_MARGIN     = 40
SNAPSHOT_MIN_WIDTH  = 400
SNAPSHOT_MIN_HEIGHT = 300

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
running   = False
frames    = {"CAM-1": None, "CAM-2": None}
frame_count = {"CAM-1": 0, "CAM-2": 0}
connected = {"CAM-1": False, "CAM-2": False}
fps_stats = {"CAM-1": 0, "CAM-2": 0}

log  = []
lock = threading.Lock()


class CameraState:
    def __init__(self):
        self.tracks     = {}
        self.cooldowns  = {}
        self.light_color = "unknown"
        self.crossed    = {}
        self.frame_n    = 0

    def is_red_light_crossed(self, tid):
        now     = time.time()
        expired = [k for k, v in self.crossed.items() if v < now]
        for k in expired:
            del self.crossed[k]
        return tid in self.crossed

    def mark_red_light_crossed(self, tid):
        self.crossed[tid] = time.time() + CROSSED_MEMORY_SECONDS


cam_state = {"CAM-1": CameraState(), "CAM-2": CameraState()}

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory('.', 'index.html')


@app.route("/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


# ─────────────────────────────────────────────
# BURST SNAPSHOT TRIGGER
# ─────────────────────────────────────────────
def trigger_burst_snapshot(frame, x1, y1, x2, y2, cam_id, plate, violation,
                            confidence, vehicle_type, vehicle_id):
    """
    Start a 5-frame burst capture for a violation.
    Returns the folder name (used as the snapshot reference in the log).
    """
    h, w = frame.shape[:2]

    # Detect color from current frame
    margin   = SNAPSHOT_MARGIN
    cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
    cx2, cy2 = min(w, x2 + margin), min(h, y2 + margin)
    roi      = frame[cy1:cy2, cx1:cx2]
    car_color = get_cached_color(vehicle_id, roi) if roi.size > 0 else "UNKNOWN"

    # Retrieve buffered pre-violation frames (up to BURST_BEFORE) + current
    pre_frames = frame_buffers[cam_id].get_recent(BURST_BEFORE)
    pre_frames.append(frame)  # current frame = violation frame

    burst = BurstCapture(
        cam_id=cam_id, plate=plate, violation=violation,
        confidence=confidence, vehicle_type=vehicle_type,
        vehicle_id=vehicle_id, car_color=car_color,
        x1=x1, y1=y1, x2=x2, y2=y2,
        pre_frames=pre_frames,
    )

    with pending_bursts_lock:
        pending_bursts[cam_id].append(burst)

    print(f"[{cam_id}] 📸 BURST started ({BURST_TOTAL} frames) | "
          f"{car_color} {vehicle_type} | Plate: {plate} | {violation}")

    return burst.folder_name


def feed_pending_bursts(cam_id: str, raw_frame: np.ndarray):
    """
    Call once per raw frame. Feeds waiting bursts with after-frames.
    Completed bursts are finalised and removed.
    """
    completed = []
    with pending_bursts_lock:
        active = list(pending_bursts[cam_id])

    for burst in active:
        done = burst.feed_after_frame(raw_frame)
        if done:
            completed.append(burst)

    # Finalise completed bursts
    for burst in completed:
        with pending_bursts_lock:
            try:
                pending_bursts[cam_id].remove(burst)
            except ValueError:
                pass
        _save_burst_metadata(burst)
        print(f"[{burst.cam_id}] ✅ BURST complete: {burst.folder_name} "
              f"({burst.saved_count} frames saved)")


def _save_burst_metadata(burst: BurstCapture):
    """Append burst metadata to the JSON file."""
    meta = burst.metadata()
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                all_meta = json.load(f)
        else:
            all_meta = []
        all_meta.append(meta)
        with open(METADATA_FILE, 'w') as f:
            json.dump(all_meta, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save metadata: {e}")


# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────
def read_plate(frame, x1, y1, x2, y2):
    if not OCR_AVAILABLE:
        return "UNKNOWN"
    bh = y2 - y1
    if bh < OCR_MIN_HEIGHT:
        return "TOO_SMALL"
    py1 = y1 + int(bh * 0.70)
    roi = frame[py1:y2, x1:x2]
    if roi.size == 0:
        return "UNKNOWN"
    try:
        if roi.shape[0] < 40 or roi.shape[1] < 100:
            roi = cv2.resize(roi, (int(roi.shape[1] * 2), int(roi.shape[0] * 2)))
        gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel   = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        res   = reader.readtext(sharpened,
                                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        texts = [r[1] for r in res if r[2] > 0.3 and len(r[1]) >= 4]
        return texts[0].upper() if texts else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def can_log(state, plate, violation):
    key = (plate, violation)
    now = time.time()
    if now - state.cooldowns.get(key, 0) < COOLDOWN_SECONDS:
        return False
    state.cooldowns[key] = now
    return True


def make_tid(cls_id, cx, cy, grid=60):
    return f"{cls_id}_{cx // grid}_{cy // grid}"


def classify_light(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"
    h      = roi.shape[0]
    third  = max(1, h // 3)
    hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    r1     = cv2.inRange(hsv, np.array([0,   120, 100]), np.array([10,  255, 255]))
    r2     = cv2.inRange(hsv, np.array([170, 120, 100]), np.array([180, 255, 255]))
    red    = cv2.bitwise_or(r1, r2)
    green  = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([90, 255, 255]))
    if cv2.countNonZero(red[:third])        > 10: return "red"
    if cv2.countNonZero(green[2*third:])    > 10: return "green"
    return "unknown"


# ─────────────────────────────────────────────
# VIOLATION ENGINE
# ─────────────────────────────────────────────
def detect_violations(detections, cam_id, frame):
    state  = cam_state[cam_id]
    fh, fw = frame.shape[:2]
    events = []
    now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    now    = time.time()

    do_scheduled_ocr = (state.frame_n % OCR_EVERY_N_FRAMES == 0)
    state.frame_n += 1

    for d in detections:
        if d['cls_id'] == CLS_TRAFFIC_LIGHT:
            c = classify_light(frame, d['x1'], d['y1'], d['x2'], d['y2'])
            if c != "unknown":
                state.light_color = c

    col = {'red': (0,0,255), 'green': (0,255,0)}.get(state.light_color, (128,128,128))
    cv2.putText(frame, f"LIGHT: {state.light_color.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    for d in detections:
        cls_id        = d['cls_id']
        x1,y1,x2,y2  = d['x1'], d['y1'], d['x2'], d['y2']
        conf          = d['conf']
        label         = d['label']
        cx, cy        = (x1+x2)//2, (y1+y2)//2
        bw, bh        = x2-x1, y2-y1

        tid  = make_tid(cls_id, cx, cy)
        prev = state.tracks.get(tid, {
            'bw': bw, 'cx': cx, 'cy': cy,
            'still_count': 0, 'still_start': None
        })

        violation  = None
        force_ocr  = False

        # ── RED LIGHT ──
        if (cls_id in VEHICLE_CLASSES and
            state.light_color == "red" and
            cy > fh * 0.40 and
            not state.is_red_light_crossed(tid)):
            if abs(cx - prev['cx']) > 5 or abs(cy - prev['cy']) > 5:
                violation = "RED_LIGHT"
                state.mark_red_light_crossed(tid)
                force_ocr = True

        # ── SPEEDING ──
        if violation is None and cls_id in VEHICLE_CLASSES:
            growth = (bw - prev['bw']) / max(prev['bw'], 1)
            if growth > SPEED_GROWTH_THRESHOLD:
                violation = "SPEEDING"
                force_ocr = True

        # ── WRONG LANE ──
        if violation is None and cls_id in VEHICLE_CLASSES:
            if (cx / fw) < WRONG_LANE_THRESHOLD and cy > fh * 0.3:
                violation = "WRONG_LANE"
                force_ocr = True

        # ── ILLEGAL PARKING ──
        if violation is None and cls_id in VEHICLE_CLASSES:
            if abs(cx - prev['cx']) < 3 and abs(cy - prev['cy']) < 3:
                prev['still_count'] += 1
                if prev['still_start'] is None:
                    prev['still_start'] = now
            else:
                prev['still_count']  = 0
                prev['still_start']  = None

            if (prev['still_count'] >= PARKING_FRAME_THRESHOLD and
                    prev['still_start'] is not None):
                still_duration = now - prev['still_start']
                if still_duration >= PARKING_MIN_DURATION:
                    if cx / fw < WRONG_LANE_THRESHOLD or state.light_color != "red":
                        violation = "ILLEGAL_PARKING"
                        force_ocr = True

        # ── PHONE USE ──
        if violation is None and cls_id == CLS_CELL_PHONE:
            violation = "PHONE_USE"
            force_ocr = True

        # Update track
        state.tracks[tid] = {
            'bw':          bw,
            'cx':          cx,
            'cy':          cy,
            'still_count': prev.get('still_count', 0),
            'still_start': prev.get('still_start', None),
        }

        # ── Handle violation ──
        if violation:
            plate = "UNKNOWN"
            if cls_id in VEHICLE_CLASSES:
                if do_scheduled_ocr or force_ocr:
                    plate = read_plate(frame, x1, y1, x2, y2)

            # ── BURST SNAPSHOT (5 frames) ──
            snap_folder = ""
            if cls_id in SNAPSHOT_CLASSES:
                vehicle_id  = f"{cam_id}_{tid}"
                snap_folder = trigger_burst_snapshot(
                    frame, x1, y1, x2, y2, cam_id,
                    plate, violation, f"{int(conf*100)}%", label, vehicle_id
                )

            if can_log(state, plate, violation):
                events.append({
                    "ts":            now_ts,
                    "camera":        cam_id,
                    "plate":         plate,
                    "vehicle_class": label,
                    "violation":     violation.replace("_", " "),
                    "probability":   f"{int(conf*100)}%",
                    "snapshot":      snap_folder,   # folder name (contains 5 frames)
                    "burst_frames":  BURST_TOTAL,
                })
                print(f"[{cam_id}] 🚨 {violation} - {plate}")

    return events


# ─────────────────────────────────────────────
# DRAW OVERLAY
# ─────────────────────────────────────────────
def draw_detections(frame, detections, cam_id):
    fh, fw = frame.shape[:2]
    lx = int(fw * WRONG_LANE_THRESHOLD)
    cv2.line(frame, (lx, 0), (lx, fh), (0, 200, 255), 2)
    cv2.putText(frame, "LANE", (lx+4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)

    fps = fps_stats.get(cam_id, 0)
    cv2.putText(frame, f"FPS: {fps}", (fw-80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    color_map = {
        CLS_CAR:           (0,  255, 100),
        CLS_TRUCK:         (0,  200, 255),
        CLS_BUS:           (255,165,  0),
        CLS_MOTORCYCLE:    (255,255,  0),
        CLS_BICYCLE:       (180,255,180),
        CLS_PERSON:        (255,100,100),
        CLS_CELL_PHONE:    (0,  100, 255),
        CLS_TRAFFIC_LIGHT: (200,200,  0),
    }

    for d in detections:
        x1,y1,x2,y2 = d['x1'], d['y1'], d['x2'], d['y2']
        col   = color_map.get(d['cls_id'], (180,180,180))
        label = f"{d['label']} {d['conf']:.0%}"
        if d.get('plate') and d['plate'] not in ['UNKNOWN','TOO_SMALL']:
            label += f" [{d['plate']}]"
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), col, -1)
        cv2.putText(frame, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return frame


# ─────────────────────────────────────────────
# PROCESS FRAME
# ─────────────────────────────────────────────
def process_frame(frame, cam_id):
    # Push raw frame into buffer BEFORE detection (for pre-violation frames)
    frame_buffers[cam_id].push(frame)

    # Feed any pending after-frames for this camera
    feed_pending_bursts(cam_id, frame)

    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id not in ALL_USED_CLASSES:
                    continue
                detections.append({
                    'x1': x1,'y1': y1,'x2': x2,'y2': y2,
                    'conf': conf,'cls_id': cls_id,
                    'label': model.names[cls_id],
                })

    events = detect_violations(detections, cam_id, frame)
    if events:
        with lock:
            log.extend(events)

    return draw_detections(frame, detections, cam_id)


# ─────────────────────────────────────────────
# ESP32-CAM STREAM FETCHER
# ─────────────────────────────────────────────
def fetch_esp32_stream(cam_id, url):
    global running, connected, frame_count, fps_stats

    print(f"[{cam_id}] Connecting to {url}")
    if not url.startswith('http'):
        url = 'http://' + url
    if not url.endswith('/stream'):
        url = url.rstrip('/') + '/stream'

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    frame_times     = deque(maxlen=30)
    reconnect_delay = 1

    while running:
        buffer          = b''
        stream          = None
        frame_count_local = 0

        try:
            req    = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            stream = urllib.request.urlopen(req, timeout=5, context=ctx)

            print(f"[{cam_id}] Connected!")
            connected[cam_id]  = True
            reconnect_delay    = 1
            buffer             = b''

            while running:
                try:
                    chunk = stream.read(4096)
                    if not chunk:
                        break

                    buffer += chunk
                    if len(buffer) > 500000:
                        buffer = buffer[-250000:]

                    start = buffer.find(b'\xff\xd8')
                    end   = buffer.find(b'\xff\xd9')

                    if start != -1 and end != -1 and end > start:
                        frame_start = time.time()
                        jpeg_data   = buffer[start:end+2]
                        buffer      = buffer[end+2:]

                        img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                        if frame is not None:
                            frame_count_local += 1
                            frame = cv2.resize(frame, (640, 480))

                            try:
                                processed = process_frame(frame, cam_id)
                            except Exception as e:
                                print(f"[{cam_id}] Process error: {e}")
                                processed = frame

                            _, buf = cv2.imencode('.jpg', processed,
                                                  [cv2.IMWRITE_JPEG_QUALITY, 70])
                            with lock:
                                frames[cam_id]        = buf.tobytes()
                                frame_count[cam_id]  += 1

                            frame_time = time.time() - frame_start
                            frame_times.append(frame_time)
                            if len(frame_times) >= 10:
                                fps_stats[cam_id] = int(len(frame_times)/sum(frame_times))

                            if frame_count_local % 100 == 0:
                                print(f"[{cam_id}] FPS: {fps_stats[cam_id]} | "
                                      f"Frames: {frame_count_local}")
                        else:
                            buffer = b''

                except socket.timeout:
                    print(f"[{cam_id}] Socket timeout, reconnecting...")
                    break
                except Exception as e:
                    print(f"[{cam_id}] Stream read error: {e}")
                    break

        except Exception as e:
            print(f"[{cam_id}] Connection error: {e}")
            connected[cam_id]  = False
            reconnect_delay    = min(reconnect_delay * 2, 10)
            time.sleep(reconnect_delay)
        finally:
            if stream:
                try: stream.close()
                except Exception: pass

    connected[cam_id] = False
    print(f"[{cam_id}] Stopped")


# ─────────────────────────────────────────────
# MJPEG STREAM GENERATOR
# ─────────────────────────────────────────────
def generate_frames(cam_id):
    while True:
        with lock:
            frame_data = frames.get(cam_id)
        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        else:
            time.sleep(0.05)


@app.route("/video/<cam_id>")
def video(cam_id):
    if cam_id not in ["CAM-1", "CAM-2"]:
        return "Invalid camera ID", 400
    response = Response(generate_frames(cam_id),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'
    return response


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/start", methods=["POST"])
def start():
    global running
    data    = request.json or {}
    cam1    = data.get("cam1", "").strip()
    cam2    = data.get("cam2", "").strip()
    running = True
    started = []
    if cam1:
        threading.Thread(target=fetch_esp32_stream, args=("CAM-1", cam1), daemon=True).start()
        started.append("CAM-1")
    if cam2:
        threading.Thread(target=fetch_esp32_stream, args=("CAM-2", cam2), daemon=True).start()
        started.append("CAM-2")
    return jsonify({"started": started})


@app.route("/api/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})


@app.route("/api/status")
def status():
    return jsonify({
        "CAM-1": {"connected": connected["CAM-1"], "frames": frame_count["CAM-1"]},
        "CAM-2": {"connected": connected["CAM-2"], "frames": frame_count["CAM-2"]},
    })


@app.route("/api/log")
def get_log():
    with lock:
        return jsonify(list(log[-500:]))


@app.route("/api/log/clear", methods=["POST"])
def clear_log():
    with lock:
        log.clear()
    return jsonify({"status": "cleared"})


@app.route("/api/snapshots")
def list_snapshots():
    """List all burst folders (each = one violation event)."""
    try:
        entries = sorted(os.listdir(SNAPSHOT_DIR), reverse=True)
        folders = [e for e in entries
                   if os.path.isdir(os.path.join(SNAPSHOT_DIR, e))]
        return jsonify({"snapshots": folders})
    except Exception as e:
        return jsonify({"snapshots": [], "error": str(e)})


@app.route("/api/snapshots/<folder>")
def list_burst_frames(folder):
    """List the individual frame files inside a burst folder."""
    folder_path = os.path.join(SNAPSHOT_DIR, folder)
    if not os.path.isdir(folder_path):
        return jsonify({"error": "Not found"}), 404
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    return jsonify({"folder": folder, "frames": files})


@app.route("/api/metadata")
def get_metadata():
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return jsonify(json.load(f))
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/scan")
def scan():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "192.168.1.1"

    subnet     = ".".join(local_ip.split(".")[:3])
    candidates = [f"{subnet}.{i}" for i in range(1, 255)]

    def probe(ip):
        for port, path in [(81, "/stream"), (80, "/stream")]:
            try:
                s = socket.create_connection((ip, port), timeout=0.3)
                s.close()
                return f"http://{ip}:{port}{path}"
            except OSError:
                pass
        return None

    found = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as ex:
        for url in ex.map(probe, candidates):
            if url:
                found.append(url)
            if len(found) >= 2:
                break

    return jsonify({"cameras": found})


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ROAD SAFETY MONITOR - 5-FRAME BURST CAPTURE ON VIOLATION")
    print("=" * 60)
    print(f"[INFO] Server:          http://localhost:5000")
    print(f"[INFO] Snapshots dir:   ./{SNAPSHOT_DIR}/")
    print(f"[INFO] Metadata file:   {METADATA_FILE}")
    print(f"[INFO] Device:          {DEVICE.upper()}")
    print(f"[INFO] Burst frames:    {BURST_BEFORE} before + 1 current + {BURST_AFTER} after = {BURST_TOTAL}")
    print(f"[INFO] Color Detection: {'ENABLED' if SKLEARN_AVAILABLE else 'LIMITED'}")
    print(f"[INFO] OCR:             {'ENABLED' if OCR_AVAILABLE else 'DISABLED'}")
    print("=" * 60)
    print("[INFO] Each violation creates a folder with 5 timestamped frames")
    print("[INFO] Press Ctrl+C to stop")
    print("=" * 60)

   if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
