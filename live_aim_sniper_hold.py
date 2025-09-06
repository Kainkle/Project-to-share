# Version: SMOOTH-2.0 (2025-09-05)
# Live Aim — Lock-First Alpha–Beta Tracker + S‑Curve Motion (Fast, Stable, Stick-First)
# - Lock-first tracker with alpha–beta filtering (predict + gentle update)
# - Strict stickiness: no switching while a valid match persists
# - Switch-hold: switch only after N consecutive detect-frame misses
# - Visibility TTL: hide overlay/aim immediately when no match
# - Smooth motion: S-curve velocity + accel limits + direction stickiness (no ping-pong)

import time
import argparse
import threading
import queue
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import math
from collections import deque

# ----- Optional input -----
try:
    import keyboard
    HAVE_KEYBOARD = True
except Exception:
    HAVE_KEYBOARD = False

# ----- Optional mouse output -----
try:
    import win32api, win32con
    HAVE_WIN32 = True
except Exception:
    HAVE_WIN32 = False

try:
    import mouse as mouse_lib
    HAVE_MOUSE_LIB = True
except Exception:
    HAVE_MOUSE_LIB = False

# ----- Optional Arduino -----
try:
    from arduino_mouse import ArduinoMouse
    HAVE_ARDUINO = True
except Exception:
    HAVE_ARDUINO = False

# ---------------- Utils ----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def box_center_wh(xyxy):
    x1, y1, x2, y2 = xyxy
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h

def cxcywh_to_xyxy(cx, cy, w, h):
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return (x1, y1, x2, y2)

def get_primary_monitor_dims():
    try:
        import mss
        with mss.mss() as sct:
            m = sct.monitors[1]
            return m["width"], m["height"]
    except Exception:
        return 1920, 1080

def center_roi(screen_w, screen_h, roi_w=480, roi_h=480):
    roi_w = min(roi_w, screen_w)
    roi_h = min(roi_h, screen_h)
    left = max(0, (screen_w - roi_w) // 2)
    top = max(0, (screen_h - roi_h) // 2)
    return left, top, roi_w, roi_h

def parse_classes_arg(model_names, classes_str):
    if classes_str.lower() == "all": return None
    name_to_idx = {v.lower(): k for k, v in model_names.items()}
    out = []
    for tok in classes_str.split(","):
        tok = tok.strip().lower()
        if not tok: continue
        try:
            out.append(int(tok)); continue
        except ValueError:
            pass
        if tok in name_to_idx:
            out.append(name_to_idx[tok])
    return sorted(list(set(out))) if out else None

# ---------------- HUD ----------------
def draw_crosshair(frame, cx, cy, deadzone_px):
    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
    if deadzone_px > 0:
        cv2.circle(frame, (cx, cy), deadzone_px, (0, 200, 0), 1)
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)

def move_mouse_relative(dx, dy, arduino=None):
    if arduino is not None:
        arduino.move(dx, dy)
        return
    if HAVE_WIN32:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)
    elif HAVE_MOUSE_LIB:
        mouse_lib.move(dx, dy, absolute=False, duration=0)

def draw_hud(frame, cap_fps, inf_ms, roi_size, det_count, aim_on, smoothing, aim_height_pct, arduino_status, target_state, mode_tip=""):
    bar_color = (60, 160, 60) if aim_on else (60, 60, 60)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), bar_color, -1)
    text = f"FPS {cap_fps:.1f} | {inf_ms:.1f} ms | ROI {roi_size[0]}x{roi_size[1]} | det {det_count} | Aim {'ON' if aim_on else 'OFF'} | Smooth {smoothing} | AimH {aim_height_pct}% | {arduino_status} | {target_state} {mode_tip}"
    cv2.putText(frame, text, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

# ---------------- One Euro (optional aimpoint micro-smoothing) ----------------
class LowPass:
    def __init__(self): self.initialized, self.y = False, 0.0
    def reset(self): self.initialized = False
    def apply(self, x, a):
        if not self.initialized:
            self.y = x; self.initialized = True; return x
        self.y = a * x + (1 - a) * self.y
        return self.y

def alpha_from_fc(fc_hz, dt):
    fc = max(1e-6, float(fc_hz)); dt = max(1e-6, float(dt))
    tau = 1.0 / (2.0 * math.pi * fc)
    return 1.0 / (1.0 + tau / dt)

class OneEuroFilter:
    def __init__(self, mincutoff=2.0, beta=0.02, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = float(mincutoff), float(beta), float(dcutoff)
        self.last_t = None; self.x_lpf = LowPass(); self.dx_lpf = LowPass(); self.prev_x = None
    def set_params(self, mincutoff=None, beta=None, dcutoff=None):
        if mincutoff is not None: self.mincutoff = float(mincutoff)
        if beta is not None: self.beta = float(beta)
        if dcutoff is not None: self.dcutoff = float(dcutoff)
    def reset(self):
        self.last_t = None; self.x_lpf.reset(); self.dx_lpf.reset(); self.prev_x = None
    def apply(self, t_now, x):
        if self.last_t is None:
            self.last_t = float(t_now); self.prev_x = float(x)
            self.x_lpf.initialized = False; self.dx_lpf.initialized = False
            return float(x)
        dt = max(1e-6, float(t_now) - self.last_t); self.last_t = float(t_now)
        dx = (float(x) - self.prev_x) / dt; self.prev_x = float(x)
        a_d = alpha_from_fc(self.dcutoff, dt); dx_hat = self.dx_lpf.apply(dx, a_d)
        cutoff = self.mincutoff + self.beta * abs(dx_hat); a_x = alpha_from_fc(cutoff, dt)
        return self.x_lpf.apply(float(x), a_x)

# ---------------- Motion controller (S-curve + accel + stickiness) ----------------
class MotionController:
    def __init__(self, vmax_pps=900.0, amax_pps2=3600.0, kv_per_s=12.0, near_band_px=4, dir_stick_ms=140):
        self.vx = 0.0; self.vy = 0.0
        self.rx = 0.0; self.ry = 0.0
        self.prev_sign_x = 0; self.prev_sign_y = 0
        self.last_flip_t_x = 0.0; self.last_flip_t_y = 0.0
        self.vmax = float(vmax_pps)
        self.amax = float(amax_pps2)
        self.kv = float(kv_per_s)
        self.near_band = int(max(0, near_band_px))
        self.dir_stick_ms = int(max(0, dir_stick_ms))

    def set_params(self, vmax_pps=None, amax_pps2=None, kv_per_s=None, near_band_px=None, dir_stick_ms=None):
        if vmax_pps is not None: self.vmax = float(vmax_pps)
        if amax_pps2 is not None: self.amax = float(amax_pps2)
        if kv_per_s is not None: self.kv = float(kv_per_s)
        if near_band_px is not None: self.near_band = int(max(0, near_band_px))
        if dir_stick_ms is not None: self.dir_stick_ms = int(max(0, dir_stick_ms))

    def reset(self):
        self.vx = 0.0; self.vy = 0.0
        self.rx = 0.0; self.ry = 0.0
        self.prev_sign_x = 0; self.prev_sign_y = 0
        self.last_flip_t_x = 0.0; self.last_flip_t_y = 0.0

    def _apply_dir_stick(self, comp_err, comp_v, prev_sign, last_flip_t, now):
        sign = 0
        if comp_v > 1e-6: sign = 1
        elif comp_v < -1e-6: sign = -1
        if sign != 0 and prev_sign != 0 and sign != prev_sign and abs(comp_err) <= self.near_band:
            if (now - last_flip_t) * 1000.0 < self.dir_stick_ms:
                return 0.0, prev_sign, last_flip_t
            else:
                last_flip_t = now
                prev_sign = sign
                return comp_v, prev_sign, last_flip_t
        if sign != 0 and sign != prev_sign:
            prev_sign = sign
            last_flip_t = now
        return comp_v, prev_sign, last_flip_t

    def step(self, err_x, err_y, dt_s, deadzone, jitter_px, step_cap, now_s):
        if abs(err_x) <= max(deadzone, jitter_px): err_x = 0.0
        if abs(err_y) <= max(deadzone, jitter_px): err_y = 0.0
        if err_x == 0.0 and err_y == 0.0:
            self.vx *= 0.5; self.vy *= 0.5
            out_x = int(round(self.rx)); out_y = int(round(self.ry))
            self.rx -= out_x; self.ry -= out_y
            return out_x, out_y

        err_mag = math.hypot(err_x, err_y)
        if err_mag > 1e-6:
            ux, uy = err_x / err_mag, err_y / err_mag
        else:
            ux, uy = 0.0, 0.0
        v_des_mag = min(self.vmax, self.kv * err_mag)
        v_des_x = v_des_mag * ux
        v_des_y = v_des_mag * uy

        dvx = v_des_x - self.vx
        dvy = v_des_y - self.vy
        dv_mag = math.hypot(dvx, dvy)
        dv_max = self.amax * dt_s
        if dv_mag > dv_max and dv_mag > 0:
            scale = dv_max / dv_mag
            dvx *= scale; dvy *= scale
        self.vx += dvx; self.vy += dvy

        self.vx, self.prev_sign_x, self.last_flip_t_x = self._apply_dir_stick(err_x, self.vx, self.prev_sign_x, self.last_flip_t_x, now_s)
        self.vy, self.prev_sign_y, self.last_flip_t_y = self._apply_dir_stick(err_y, self.vy, self.prev_sign_y, self.last_flip_t_y, now_s)

        sx_f = self.vx * dt_s + self.rx
        sy_f = self.vy * dt_s + self.ry

        sx_f = math.copysign(min(abs(sx_f), abs(err_x)), err_x)
        sy_f = math.copysign(min(abs(sy_f), abs(err_y)), err_y)

        mag_f = math.hypot(sx_f, sy_f)
        err_mag = math.hypot(err_x, err_y)
        vcap = min(step_cap, err_mag)
        if mag_f > vcap and mag_f > 0:
            scale = vcap / mag_f
            sx_f *= scale; sy_f *= scale

        out_x = int(round(sx_f)); out_y = int(round(sy_f))
        self.rx = sx_f - out_x; self.ry = sy_f - out_y
        return out_x, out_y

# ---------------- Smoothing map (for motion params) ----------------
def map_smoothing(level):
    s = clamp(level, 0, 100)
    if s <= 5:
        return {"mode":"snap","deadzone":0,"jitter_px":0,"step_cap":10000}
    t = (s - 6) / 94.0
    deadzone = int(round(1 * (1 - t) + 6 * t))
    jitter_px = int(round(1 * (1 - t) + 3 * t))
    step_cap = int(round(60 * (1 - t) + 10 * t))
    return {"mode":"smooth","deadzone":deadzone,"jitter_px":jitter_px,"step_cap":step_cap}

# ---------------- Scoring ----------------
def compute_scores_all(boxes, confs, cx, cy, roi_w, roi_h, min_box_h):
    if boxes.size == 0:
        return np.empty((0,), dtype=float)
    centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) * 0.5,
                               (boxes[:, 1] + boxes[:, 3]) * 0.5])
    dists = np.linalg.norm(centers - np.array([cx, cy]), axis=1)
    heights = (boxes[:, 3] - boxes[:, 1])
    widths  = (boxes[:, 2] - boxes[:, 0])
    aspect  = heights / np.maximum(1.0, widths)
    area    = (heights * widths) / max(1.0, roi_w * roi_h)
    scores = confs.copy()
    scores -= 0.0025 * dists
    scores[heights < min_box_h] -= 10.0
    scores += 0.20 * np.clip(aspect - 1.0, -0.8, 1.0)
    scores += 0.12 * area
    return scores

# ---------------- Alpha–Beta tracker (single target) ----------------
class AlphaBetaBoxTracker:
    def __init__(self, alpha_pos=0.4, beta_pos=0.8, alpha_size=0.2, beta_size=0.4):
        self.cx = None; self.cy = None; self.w = None; self.h = None
        self.vx = 0.0; self.vy = 0.0; self.vw = 0.0; self.vh = 0.0
        self.alpha_pos = float(alpha_pos)
        self.beta_pos  = float(beta_pos)
        self.alpha_size = float(alpha_size)
        self.beta_size  = float(beta_size)

    def reset(self):
        self.cx = self.cy = self.w = self.h = None
        self.vx = self.vy = self.vw = self.vh = 0.0

    def is_initialized(self):
        return self.cx is not None

    def init_from_box(self, xyxy):
        cx, cy, w, h = box_center_wh(xyxy)
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.vx = self.vy = self.vw = self.vh = 0.0

    def predict(self, dt):
        if not self.is_initialized():
            return None
        cxp = self.cx + self.vx * dt
        cyp = self.cy + self.vy * dt
        wp  = max(1.0, self.w + self.vw * dt)
        hp  = max(1.0, self.h + self.vh * dt)
        return (cxp, cyp, wp, hp)

    def update(self, dt, meas_xyxy):
        mcx, mcy, mw, mh = box_center_wh(meas_xyxy)
        if not self.is_initialized():
            self.init_from_box(meas_xyxy)
            return
        # Predict
        cxp = self.cx + self.vx * dt
        cyp = self.cy + self.vy * dt
        wp  = max(1.0, self.w + self.vw * dt)
        hp  = max(1.0, self.h + self.vh * dt)
        # Residuals
        rx = mcx - cxp; ry = mcy - cyp
        rw = mw  - wp;  rh = mh  - hp
        # Update
        a_pos, b_pos = self.alpha_pos, self.beta_pos
        a_s, b_s    = self.alpha_size, self.beta_size
        self.cx = cxp + a_pos * rx
        self.cy = cyp + a_pos * ry
        self.w  = max(1.0, wp  + a_s   * rw)
        self.h  = max(1.0, hp  + a_s   * rh)
        # Velocity updates
        if dt > 1e-4:
            self.vx = self.vx + (b_pos * rx) / dt
            self.vy = self.vy + (b_pos * ry) / dt
            self.vw = self.vw + (b_s   * rw) / dt
            self.vh = self.vh + (b_s   * rh) / dt

    def xyxy(self):
        if not self.is_initialized():
            return None
        return cxcywh_to_xyxy(self.cx, self.cy, self.w, self.h)

# ---------------- Async YOLO inference ----------------
class InferenceWorker:
    def __init__(self, model, device, conf, iou, imgsz, half, max_det, class_filter):
        self.model = model; self.device = device; self.conf = conf; self.iou = iou
        self.imgsz = imgsz; self.half = half; self.max_det = max_det; self.class_filter = class_filter
        self.q = queue.Queue(maxsize=1); self.lock = threading.Lock(); self.latest = None
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()
    def submit(self, frame_bgr):
        try: _ = self.q.get_nowait()
        except Exception: pass
        self.q.put(frame_bgr, block=False)
    def get_latest(self):
        with self.lock:
            return self.latest
    def shutdown(self):
        self.stop_evt.set()
        try: self.q.put_nowait(None)
        except Exception: pass
        try: self.thread.join(timeout=0.5)
        except Exception: pass
    def _loop(self):
        while not self.stop_evt.is_set():
            try: frame_bgr = self.q.get(timeout=0.2)
            except queue.Empty: continue
            if frame_bgr is None: break
            t0 = time.time()
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = self.model.predict(
                    frame_rgb, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                    device=self.device, half=(self.device == "cuda" and self.half),
                    max_det=self.max_det, verbose=False
                )[0]
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.half:
                    print("[Worker] CUDA OOM with FP16; falling back to FP32."); self.half = False; continue
                else:
                    print(f"[Worker] RuntimeError: {e}"); continue
            boxes_all = res.boxes.xyxy.cpu().numpy() if res.boxes.shape[0] > 0 else np.empty((0, 4), dtype=float)
            confs_all = res.boxes.conf.cpu().numpy() if res.boxes.shape[0] > 0 else np.empty((0,), dtype=float)
            if self.class_filter is not None and res.boxes.shape[0] > 0:
                cl_all = res.boxes.cls.cpu().numpy().astype(int)
                mask = np.isin(cl_all, self.class_filter)
                boxes_all = boxes_all[mask]; confs_all = confs_all[mask]
            inf_ms = (time.time() - t0) * 1000.0
            with self.lock:
                self.latest = (boxes_all, confs_all, inf_ms)

# ---------------- Capture helpers ----------------
def try_dxcam_async(output_idx):
    try:
        import dxcam
    except Exception as e:
        print(f"[DXCam] Import failed: {e}")
        return None, "import failed"
    try:
        cam = dxcam.create(output_idx=output_idx, output_color="BGR")
        if cam is None: return None, "create failed"
        cam.start(target_fps=60, video_mode=True)
        time.sleep(0.25)
        for _ in range(60):
            f = cam.get_latest_frame()
            if f is not None:
                return cam, None
            time.sleep(0.01)
        cam.stop(); cam.release()
        return None, "no frames"
    except Exception as e:
        return None, str(e)

def run_dxcam_loop(cam, roi_xyxy, on_frame):
    cap_fps, frames, last = 0.0, 0, time.time()
    while True:
        frame = cam.get_latest_frame()
        if frame is None:
            time.sleep(0.005); continue
        frame = np.ascontiguousarray(frame)
        l, t, r, b = roi_xyxy
        frame = frame[int(t):int(b), int(l):int(r)]
        frames += 1
        now = time.time()
        if now - last >= 1.0:
            cap_fps = frames / (now - last); frames, last = 0, now
        if on_frame(frame, cap_fps) is False:
            break
    try: cam.stop()
    except Exception: pass
    try: cam.release()
    except Exception: pass

def run_mss_loop(roi_xywh, on_frame):
    try:
        import mss
    except Exception as e:
        print(f"[MSS] Import failed: {e}")
        return False
    left, top, w, h = roi_xywh
    mon = {"left": int(left), "top": int(top), "width": int(w), "height": int(h)}
    cap_fps, frames, last = 0.0, 0, time.time()
    with mss.mss() as sct:
        while True:
            shot = sct.grab(mon)
            frame = np.ascontiguousarray(np.array(shot)[:, :, :3])
            frames += 1
            now = time.time()
            if now - last >= 1.0:
                cap_fps = frames / (now - last); frames, last = 0, now
            if on_frame(frame, cap_fps) is False:
                break
    return True

# ---------------- GUI ----------------
CTRL_WIN = "Aim Controls"
def create_controls(initial_smoothing=14, initial_aimh=30):
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_AUTOSIZE)
    def _noop(_): pass
    cv2.createTrackbar("Smoothing", CTRL_WIN, int(clamp(initial_smoothing, 0, 100)), 100, _noop)
    cv2.createTrackbar("Aim height %", CTRL_WIN, int(clamp(initial_aimh, 0, 100)), 100, _noop)

def read_controls():
    smoothing = cv2.getTrackbarPos("Smoothing", CTRL_WIN)
    aim_height_pct = cv2.getTrackbarPos("Aim height %", CTRL_WIN)
    return smoothing, aim_height_pct

# ---------------- Debouncer ----------------
class HoldDebouncer:
    def __init__(self, on_ms=60, off_ms=150):
        self.on_ms = max(0, int(on_ms)); self.off_ms = max(0, int(off_ms))
        self.state = False; self._acc_true = 0.0; self._acc_false = 0.0
    def update(self, raw: bool, dt_ms: float) -> bool:
        if raw: self._acc_true += dt_ms; self._acc_false = 0.0
        else:   self._acc_false += dt_ms; self._acc_true = 0.0
        if not self.state and self._acc_true >= self.on_ms:
            self.state = True; self._acc_true = 0.0
        elif self.state and self._acc_false >= self.off_ms:
            self.state = False; self._acc_false = 0.0
        return self.state

# ---------------- Main ----------------
def main():
    print("=== Live Aim — Lock-First Alpha–Beta | Version: SMOOTH-2.0 (2025-09-05) ===")
    parser = argparse.ArgumentParser(description="Live YOLOv8 Aim — Lock-First Alpha–Beta Tracker + S-Curve Motion")
    parser.add_argument("--mode", choices=["auto", "dxcam", "mss"], default="mss")
    parser.add_argument("--output-idx", type=int, default=-1)
    parser.add_argument("--roi-w", type=int, default=416)
    parser.add_argument("--roi-h", type=int, default=416)
    parser.add_argument("--imgsz", type=int, default=352)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--classes", type=str, default="person")
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--max-det", type=int, default=6)
    parser.add_argument("--try-half", action="store_true")

    # GUI defaults
    parser.add_argument("--smoothing", type=int, default=14)
    parser.add_argument("--aim-height", type=int, default=30)

    # Hold/debounce
    parser.add_argument("--hold-key", type=str, default="n")
    parser.add_argument("--hold-on-ms", type=int, default=60)
    parser.add_argument("--hold-off-ms", type=int, default=150)

    # Selection/stability
    parser.add_argument("--min-box-h", type=int, default=64)
    parser.add_argument("--lock-class", action="store_true")

    # Detector throttle
    parser.add_argument("--det-every", type=int, default=2, help="Run detection every N frames")

    # Stickiness & switching
    parser.add_argument("--gate-dist-px", type=float, default=64.0)
    parser.add_argument("--gate-iou", type=float, default=0.45)
    parser.add_argument("--switch-hold", type=int, default=4, help="Consecutive detect-frame misses before switching targets")
    parser.add_argument("--display-ttl-ms", type=int, default=220)
    parser.add_argument("--unlock-ttl-ms", type=int, default=600)

    # Alpha–Beta gains
    parser.add_argument("--ab-alpha-pos", type=float, default=0.45)
    parser.add_argument("--ab-beta-pos", type=float, default=0.90)
    parser.add_argument("--ab-alpha-size", type=float, default=0.25)
    parser.add_argument("--ab-beta-size", type=float, default=0.50)

    # One Euro (aimpoint micro-smoothing)
    parser.add_argument("--euro", choices=["on","off"], default="on")
    parser.add_argument("--euro-mincut", type=float, default=1.8)
    parser.add_argument("--euro-beta", type=float, default=0.02)
    parser.add_argument("--euro-dcut", type=float, default=1.0)

    # Motion (per-second units)
    parser.add_argument("--vmax-pps", type=float, default=900.0)
    parser.add_argument("--amax-pps2", type=float, default=3600.0)
    parser.add_argument("--kv-per-s", type=float, default=12.0)
    parser.add_argument("--near-band-px", type=int, default=4)
    parser.add_argument("--dir-stick-ms", type=int, default=140)

    # Debug
    parser.add_argument("--draw-dets", action="store_true")

    # Arduino
    parser.add_argument("--arduino-port", type=str, default="")
    parser.add_argument("--arduino-baud", type=int, default=115200)
    args = parser.parse_args()

    if not HAVE_KEYBOARD:
        print("The 'keyboard' package is required. Install with: pip install keyboard")
        return

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # Screen/ROI
    screen_w, screen_h = get_primary_monitor_dims()
    left, top, roi_w, roi_h = center_roi(screen_w, screen_h, args.roi_w, args.roi_h)
    roi_xyxy = (left, top, left + roi_w, top + roi_h)
    roi_xywh = (left, top, roi_w, roi_h)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Device={device} | Model={args.model}")
    model = YOLO(args.model)
    names = model.names
    class_filter = parse_classes_arg(names, args.classes)

    # Arduino
    arduino = None
    if args.arduino_port and HAVE_ARDUINO:
        try:
            arduino = ArduinoMouse(args.arduino_port, args.arduino_baud)
            arduino.connect(); print(f"[Arduino] Connected on {args.arduino_port}")
        except Exception as e:
            print(f"[Arduino] Could not open {args.arduino_port}: {e}")
            arduino = None

    # GUI
    create_controls(args.smoothing, args.aim_height)
    print("[Init] GUI 'Aim Controls' created. Press and hold key to aim:", args.hold_key)

    # Debounce + async worker
    debouncer = HoldDebouncer(args.hold_on_ms, args.hold_off_ms)
    prev_debounced = False
    last_time = time.time()
    dt_s_smoothed = 0.05  # ~20 FPS guess
    inf_ms_display = 0.0

    worker = InferenceWorker(model=model, device=device, conf=args.conf, iou=args.iou,
                             imgsz=args.imgsz, half=args.try_half, max_det=args.max_det,
                             class_filter=class_filter)

    detection_on = True
    aim_on = False

    # Aimpoint micro-smoothing
    euro_x = OneEuroFilter(args.euro_mincut, args.euro_beta, args.euro_dcut)
    euro_y = OneEuroFilter(args.euro_mincut, args.euro_beta, args.euro_dcut)
    use_euro = (args.euro == "on")

    # Motion controller
    motion = MotionController(vmax_pps=args.vmax_pps, amax_pps2=args.amax_pps2, kv_per_s=args.kv_per_s,
                              near_band_px=args.near_band_px, dir_stick_ms=args.dir_stick_ms)

    # Stability state
    frame_idx = 0
    cand_hist = deque(maxlen=3)  # for pre-lock persistence (2/3 default)
    locked = False
    tracker = AlphaBetaBoxTracker(args.ab_alpha_pos, args.ab_beta_pos, args.ab_alpha_size, args.ab_beta_size)
    last_match_time = 0.0
    visible = False
    switch_miss_count = 0

    print("Controls: ESC exit | D toggle detection | Arrows move ROI | +/- resize ROI | R recenter")

    def on_frame(frame_bgr, cap_fps_local):
        nonlocal left, top, roi_w, roi_h, roi_xyxy, roi_xywh, detection_on, aim_on, prev_debounced, last_time, dt_s_smoothed, inf_ms_display
        nonlocal frame_idx, cand_hist, locked, tracker, last_match_time, visible, switch_miss_count

        frame_idx += 1
        smoothing, aim_height_pct = read_controls()
        map_params = map_smoothing(smoothing)

        # Detection throttle
        detect_this = (frame_idx == 1) or (frame_idx % max(1, args.det_every) == 0)
        if detection_on and detect_this:
            worker.submit(frame_bgr)

        now = time.time()
        dt_s = max(1e-4, now - last_time)
        last_time = now
        dt_s_smoothed = 0.85 * dt_s_smoothed + 0.15 * dt_s

        raw = keyboard.is_pressed(args.hold_key)
        deb = debouncer.update(raw, dt_s_smoothed * 1000.0)
        if deb != prev_debounced:
            aim_on = deb
            if not deb:
                motion.reset()
            if arduino:
                if deb: arduino.arm(); arduino.start_keepalive(4.0)
                else:   arduino.stop_keepalive(); arduino.disarm()
            prev_debounced = deb

        h, w = frame_bgr.shape[:2]
        cx, cy = w // 2, h // 2
        det_count = 0
        target_state = "idle"
        target_pt = None

        # Process detections on detect frames
        if detection_on and detect_this:
            latest = worker.get_latest()
            if latest is None:
                boxes = np.empty((0, 4)); confs = np.empty((0,)); inf_ms = 0.0
            else:
                boxes, confs, inf_ms = latest
            inf_ms_display = inf_ms

            if boxes.size > 0:
                det_count = boxes.shape[0]
                # Score all
                scores = compute_scores_all(boxes, confs, cx, cy, roi_w, roi_h, args.min_box_h)
                best_idx = int(np.argmax(scores)); best_box = tuple(map(float, boxes[best_idx]))
                # Pre-lock persistence
                if not locked:
                    if len(cand_hist) == 0:
                        cand_hist.append(1)
                    else:
                        # Compare best this frame with previous best by IoU
                        prev_best_xyxy = tracker.xyxy() if tracker.is_initialized() else best_box
                        cand_hist.append(1 if iou_xyxy(best_box, prev_best_xyxy) >= 0.25 else 0)
                    if sum(cand_hist) >= 2:  # 2 of last 3
                        locked = True
                        tracker.init_from_box(best_box)
                        last_match_time = now
                        switch_miss_count = 0
                        target_state = "lock"
                    else:
                        target_state = "seeking"
                else:
                    # While locked: predict, then match by IoU or proximity (stick-first)
                    pred = tracker.predict(dt_s) or tracker.xyxy()
                    cur_xyxy = cxcywh_to_xyxy(*pred) if pred and len(pred)==4 else tracker.xyxy()
                    if cur_xyxy is None:
                        cur_xyxy = best_box
                    cur_cx, cur_cy, _, _ = box_center_wh(cur_xyxy)
                    dists = np.array([math.hypot(((b[0]+b[2])*0.5) - cur_cx, ((b[1]+b[3])*0.5) - cur_cy) for b in boxes])
                    ious = np.array([iou_xyxy(cur_xyxy, b) for b in boxes])
                    gate_mask = (ious >= float(args.gate_iou)) | (dists <= float(args.gate_dist_px))
                    valid_idxs = np.where(gate_mask)[0]
                    if valid_idxs.size > 0:
                        idx_match = int(valid_idxs[np.argmax(ious[valid_idxs])])
                        tracker.update(dt_s, tuple(map(float, boxes[idx_match])))
                        last_match_time = now
                        switch_miss_count = 0
                        target_state = "track-stick"
                    else:
                        # No acceptable match this detect frame
                        switch_miss_count += 1
                        target_state = f"track-miss{switch_miss_count}"
                        # Only switch after sustained misses; otherwise keep predicting
                        if switch_miss_count >= max(1, args.switch_hold):
                            # Switch to best if far enough (clearly different)
                            if iou_xyxy(cur_xyxy, best_box) < 0.25:
                                tracker.init_from_box(best_box)
                                last_match_time = now
                                switch_miss_count = 0
                                target_state = "switch"

        # Visibility & unlock logic every frame
        if locked:
            ms_since_match = (now - last_match_time) * 1000.0
            visible = (ms_since_match <= args.display_ttl_ms)
            if ms_since_match >= args.unlock_ttl_ms:
                locked = False
                tracker.reset()
                cand_hist.clear()
                visible = False
                switch_miss_count = 0
                target_state = "unlock-ttl"
        else:
            visible = False

        # Build aimpoint if visible
        if locked and visible and tracker.is_initialized():
            bx = tracker.xyxy()
            scx, scy, sw, sh = box_center_wh(bx)
            tx = scx
            ty = bx[1] + (aim_height_pct / 100.0) * (bx[3] - bx[1])
            if use_euro:
                tx = euro_x.apply(now, tx)
                ty = euro_y.apply(now, ty)
            target_pt = (tx, ty)
            # Draw stabilized box and aimpoint
            cv2.rectangle(frame_bgr, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 255, 0), 2)
            cv2.circle(frame_bgr, (int(tx), int(ty)), 4, (0, 255, 0), -1)

        # Draw UI
        draw_crosshair(frame_bgr, cx, cy, map_params["deadzone"] if map_params["mode"] == "smooth" else 0)
        if target_pt is not None:
            cv2.arrowedLine(frame_bgr, (cx, cy), (int(target_pt[0]), int(target_pt[1])), (0, 255, 255), 2, tipLength=0.25)
        tip = f"| det@{args.det_every} gate_iou={args.gate_iou:.2f} hold={args.switch_hold}"
        draw_hud(frame_bgr, cap_fps_local, inf_ms_display, (roi_w, roi_h), det_count, aim_on, smoothing, aim_height_pct,
                 "Arduino: ON" if (HAVE_ARDUINO and arduino is not None) else "Arduino: OFF", target_state, mode_tip=tip)

        cv2.imshow("Live Aim (Lock-First + Alpha-Beta)", frame_bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cv2.destroyAllWindows(); return False
        elif key in (ord('d'), ord('D')):
            detection_on = not detection_on
        elif key in (ord('r'), ord('R')):
            left, top, roi_w, roi_h = center_roi(screen_w, screen_h, args.roi_w, args.roi_h)
        elif key == 0x250000:
            left = clamp(left - 32, 0, max(0, screen_w - roi_w))
        elif key == 0x260000:
            top = clamp(top - 32, 0, max(0, screen_h - roi_h))
        elif key == 0x270000:
            left = clamp(left + 32, 0, max(0, screen_w - roi_w))
        elif key == 0x280000:
            top = clamp(top + 32, 0, max(0, screen_h - roi_h))
        elif key in (ord('+'), ord('=')):
            roi_w = clamp(roi_w + 64, 64, screen_w); roi_h = clamp(roi_h + 64, 64, screen_h)
        elif key in (ord('-'), ord('_')):
            roi_w = clamp(roi_w - 64, 64, screen_w); roi_h = clamp(roi_h - 64, 64, screen_h)

        # Cursor move
        if aim_on and target_pt is not None:
            err_x = target_pt[0] - cx
            err_y = target_pt[1] - cy
            if map_params["mode"] == "snap":
                mdx, mdy = int(round(err_x)), int(round(err_y))
            else:
                mdx, mdy = motion.step(
                    err_x, err_y,
                    dt_s=dt_s_smoothed,
                    deadzone=map_params["deadzone"],
                    jitter_px=map_params["jitter_px"],
                    step_cap=map_params["step_cap"],
                    now_s=now
                )
            if mdx != 0 or mdy != 0:
                move_mouse_relative(mdx, mdy, arduino=arduino)

        return True

    # Launch capture
    tried_any = False
    if args.mode in ("auto", "dxcam"):
        tried_any = True
        output_candidates = [args.output_idx] if args.output_idx >= 0 else [0, 1, 2, 3]
        dx_ok = False
        for idx in output_candidates:
            cam, err = try_dxcam_async(idx)
            if cam is not None:
                print(f"[DXCam] Using output_idx={idx}")
                run_dxcam_loop(cam, roi_xyxy, on_frame)
                dx_ok = True
                break
            else:
                print(f"[DXCam] output_idx={idx} failed: {err}")
        if not dx_ok:
            print("[DXCam] Falling back to MSS...")
            ok = run_mss_loop(roi_xywh, on_frame)
            if ok:
                worker.shutdown()
                if arduino: arduino.close()
                return

    if args.mode in ("auto", "mss"):
        tried_any = True
        ok = run_mss_loop(roi_xywh, on_frame)
        if ok:
            worker.shutdown()
            if arduino: arduino.close()
            return

    if not tried_any:
        print("[Error] No capture attempted.")
    else:
        print("[Error] No capture method worked.")
    worker.shutdown()
    if arduino: arduino.close()

if __name__ == "__main__":
    main()