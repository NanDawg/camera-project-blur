"""
Gesture Blur Camera
====================
Real-time webcam app that blurs the video feed when a PEACE (✌️) hand
gesture is detected, using the MediaPipe Tasks Hand Landmarker API.

Run:
    python main.py

Requires hand_landmarker.task to be present in the same folder as this file.
See README.md for full setup instructions and hotkeys.
"""

import os
import sys
import time
import math
from datetime import datetime

import cv2
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "hand_landmarker.task"

DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720

TARGET_CAM_WIDTH = 1920
TARGET_CAM_HEIGHT = 1080

WINDOW_NAME = "Gesture Blur Camera"

# Anti-flicker with fast-attack / slow-release hysteresis:
# blur turns ON almost immediately once PEACE is seen (few frames),
# but only turns OFF after several consecutive non-PEACE frames.
# This avoids the "have to pass through open-hand first" feeling while
# still preventing blur from flickering off on a single bad frame.
GESTURE_ON_FRAMES = 2
GESTURE_OFF_FRAMES = 5

# Max Gaussian kernel size per blur mode (must end up odd at runtime).
BLUR_LEVELS = {
    "light": 15,
    "medium": 35,
    "heavy": 65,
}
DEFAULT_BLUR_MODE = "heavy"

# How fast the blur amount interpolates toward its target each frame.
# 0.0 = never moves, 1.0 = instant (no smoothing).
BLUR_SMOOTH_SPEED = 0.8

# Gesture sensitivity thresholds (tunable at runtime with '[' and ']').
FINGER_EXTENSION_THRESHOLD = 0.12
THUMB_EXTENSION_THRESHOLD = 0.06

SCREENSHOT_DIR = "screenshots"
RECORDING_DIR = "recordings"

# Best-effort raw key codes reported by cv2.waitKeyEx() for the F11 key.
# These vary across OS/OpenCV builds, so 'F' is provided as a guaranteed
# fallback fullscreen toggle regardless of platform.
F11_KEYCODES = {7405568, 122, 1113133}

# Hand landmark indices (MediaPipe Hand Landmarker - 21 points per hand)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


# ============================================================
# STARTUP CHECKS
# ============================================================

def check_model_file():
    if not os.path.isfile(MODEL_PATH):
        print("=" * 60)
        print("ERROR: hand_landmarker.task not found.")
        print(f"Expected location: {os.path.abspath(MODEL_PATH)}")
        print()
        print("The file 'hand_landmarker.task' must be placed in the")
        print("same folder as main.py (the project root).")
        print()
        print("Download it from:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")
        print("=" * 60)
        sys.exit(1)


# ============================================================
# MEDIAPIPE TASKS SETUP (new Tasks API, NOT mp.solutions.hands)
# ============================================================

def create_hand_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


# ============================================================
# GESTURE DETECTION
# ============================================================

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _is_finger_extended(landmarks, tip_idx, pip_idx, mcp_idx):
    """Rotation-tolerant extension check: compares how far the tip sits
    from the wrist relative to the pip joint, scaled by palm size."""
    wrist = landmarks[WRIST]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]

    palm_scale = _dist(wrist, mcp) + 1e-6
    tip_dist = _dist(wrist, tip)
    pip_dist = _dist(wrist, pip)

    return (tip_dist - pip_dist) / palm_scale > FINGER_EXTENSION_THRESHOLD


def _is_thumb_extended(landmarks):
    tip = landmarks[THUMB_TIP]
    ip = landmarks[THUMB_IP]
    index_mcp = landmarks[INDEX_MCP]

    palm_scale = _dist(landmarks[WRIST], index_mcp) + 1e-6
    tip_dist = _dist(index_mcp, tip)
    ip_dist = _dist(index_mcp, ip)

    return (tip_dist - ip_dist) / palm_scale > THUMB_EXTENSION_THRESHOLD


def detect_gesture(hand_landmarks):
    """
    hand_landmarks: list of 21 landmark objects (with .x, .y, .z) for ONE hand,
    as returned by HandLandmarkerResult.hand_landmarks[i].

    Returns one of: "peace", "open", "fist", "one", "thumbs_up", "unknown"
    or None if no landmarks were passed in (no hand detected).
    """
    if not hand_landmarks:
        return None

    index_ext = _is_finger_extended(hand_landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle_ext = _is_finger_extended(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_ext = _is_finger_extended(hand_landmarks, RING_TIP, RING_PIP, RING_MCP)
    pinky_ext = _is_finger_extended(hand_landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP)
    thumb_ext = _is_thumb_extended(hand_landmarks)

    fingers = (thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext)

    # PEACE ✌️ : index + middle extended, ring + pinky folded
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "peace"

    # OPEN HAND ✋ : all four main fingers extended
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "open"

    # FIST 👊 : nothing extended
    if not any(fingers):
        return "fist"

    # ONE FINGER ☝️ : only index extended
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        return "one"

    # THUMBS UP 👍 : only thumb extended
    if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "thumbs_up"

    return "unknown"


class GestureStabilizer:
    """Fast-attack / slow-release debounce for blur activation.

    - PEACE activates blur almost immediately (after `on_frames` consecutive
      PEACE reads) so there's no lag waiting for other gestures to "clear".
    - Blur only deactivates after `off_frames` consecutive non-PEACE reads,
      so a single misread frame doesn't cause flicker.

    `raw_gesture` (unsmoothed, per-frame) is still shown directly in the UI
    for a responsive label; only the *blur* decision goes through hysteresis.
    """

    def __init__(self, on_frames=GESTURE_ON_FRAMES, off_frames=GESTURE_OFF_FRAMES):
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.on_streak = 0
        self.off_streak = 0
        self.blur_active = False

    def update(self, raw_gesture):
        if raw_gesture == "peace":
            self.on_streak += 1
            self.off_streak = 0
        else:
            self.off_streak += 1
            self.on_streak = 0

        if not self.blur_active and self.on_streak >= self.on_frames:
            self.blur_active = True
        elif self.blur_active and self.off_streak >= self.off_frames:
            self.blur_active = False

        return self.blur_active


# ============================================================
# BLUR CONTROLLER (smooth interpolated transition)
# ============================================================

class BlurController:
    def __init__(self, mode=DEFAULT_BLUR_MODE):
        self.mode = mode
        self.target_strength = 0.0    # 0..1
        self.current_strength = 0.0   # 0..1, eases toward target_strength

    def set_mode(self, mode):
        if mode in BLUR_LEVELS:
            self.mode = mode

    def set_active(self, active: bool):
        self.target_strength = 1.0 if active else 0.0

    def update(self):
        diff = self.target_strength - self.current_strength
        self.current_strength += diff * BLUR_SMOOTH_SPEED
        if abs(self.current_strength) < 0.001:
            self.current_strength = 0.0
        elif abs(1.0 - self.current_strength) < 0.001:
            self.current_strength = 1.0
        return self.current_strength

    def apply(self, frame):
        strength = self.current_strength
        if strength <= 0.001:
            return frame

        max_kernel = BLUR_LEVELS[self.mode]
        kernel = int(max_kernel * strength)
        kernel = kernel if kernel % 2 == 1 else kernel + 1
        kernel = max(kernel, 1)
        if kernel <= 1:
            return frame

        return cv2.GaussianBlur(frame, (kernel, kernel), 0)


# ============================================================
# ASPECT-RATIO PRESERVING RESIZE + CENTER CROP
# ============================================================

def resize_and_crop_to_window(frame, target_w, target_h):
    """
    Scales `frame` so it fully covers a (target_w x target_h) area
    without distortion, then center-crops the overflow. Never stretches.
    """
    h, w = frame.shape[:2]
    if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
        return frame

    frame_aspect = w / h
    target_aspect = target_w / target_h

    if frame_aspect > target_aspect:
        # frame is relatively wider than the window -> match height, crop width
        new_h = target_h
        new_w = max(1, int(round(new_h * frame_aspect)))
    else:
        # frame is relatively taller than the window -> match width, crop height
        new_w = target_w
        new_h = max(1, int(round(new_w / frame_aspect)))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x_start = max(0, (new_w - target_w) // 2)
    y_start = max(0, (new_h - target_h) // 2)
    cropped = resized[y_start:y_start + target_h, x_start:x_start + target_w]
    return cropped


# ============================================================
# UI DRAWING HELPERS
# ============================================================

def draw_rounded_panel(img, x, y, w, h, radius=12, color=(20, 20, 20), alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    cv2.circle(overlay, (x + radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + w - radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + radius, y + h - radius), radius, color, -1)
    cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


GESTURE_LABELS = {
    "peace": "PEACE",
    "open": "OPEN HAND",
    "fist": "FIST",
    "one": "ONE FINGER",
    "thumbs_up": "THUMBS UP",
    "unknown": "UNKNOWN",
    None: "NORMAL",
}


# ============================================================
# MAIN APPLICATION
# ============================================================

class GestureBlurCameraApp:
    def __init__(self):
        check_model_file()

        self.cap = self._open_camera()
        self.landmarker = create_hand_landmarker()

        self.stabilizer = GestureStabilizer()
        self.blur_ctrl = BlurController()

        self.mirror = True
        self.show_ui = True
        self.is_fullscreen = False
        self.manual_mode = False
        self.manual_blur_on = False

        self.recording = False
        self.video_writer = None

        self.window_w = DEFAULT_WINDOW_WIDTH
        self.window_h = DEFAULT_WINDOW_HEIGHT

        self.fps = 0.0
        self._fps_last_time = time.time()
        self._fps_frame_counter = 0

        self._start_time_ms = int(time.time() * 1000)
        self._last_timestamp = -1

        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        os.makedirs(RECORDING_DIR, exist_ok=True)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.window_w, self.window_h)

    # ---------------- Camera setup ----------------

    def _open_camera(self):
        # CAP_DSHOW is the recommended backend on Windows for fast, reliable opens.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("=" * 60)
            print("ERROR: Camera not found.")
            print("Please check that:")
            print(" - Your webcam is connected properly")
            print(" - No other application is currently using the camera")
            print(" - Camera permissions are enabled for this app")
            print("=" * 60)
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_CAM_HEIGHT)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_w}x{actual_h}")

        ok, _ = cap.read()
        if not ok:
            print("ERROR: Camera opened but failed to deliver frames.")
            print("It might be in use by another application.")
            cap.release()
            sys.exit(1)

        return cap

    # ---------------- Main loop ----------------

    def run(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("WARNING: Failed to read frame from camera. Retrying...")
                    time.sleep(0.05)
                    continue

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int(time.time() * 1000) - self._start_time_ms
                if timestamp_ms <= self._last_timestamp:
                    timestamp_ms = self._last_timestamp + 1
                self._last_timestamp = timestamp_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                raw_gesture = None
                if result.hand_landmarks:
                    raw_gesture = detect_gesture(result.hand_landmarks[0])

                confirmed_blur_active = self.stabilizer.update(raw_gesture)

                # Gesture detection always keeps running; manual mode only
                # overrides what drives the *blur output*.
                if not self.manual_mode:
                    self.blur_ctrl.set_active(confirmed_blur_active)
                else:
                    self.blur_ctrl.set_active(self.manual_blur_on)

                self.blur_ctrl.update()
                processed = self.blur_ctrl.apply(frame)

                self._update_fps()

                win_w, win_h = self.get_window_size()
                display_frame = resize_and_crop_to_window(processed, win_w, win_h)
                screenshot_source = display_frame.copy()

                if self.recording and self.video_writer is not None:
                    self.video_writer.write(processed)

                if self.show_ui:
                    self._draw_overlay(display_frame, raw_gesture)

                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKeyEx(1)
                if key != -1:
                    if not self._handle_key(key, screenshot_source):
                        break
        finally:
            self._cleanup()

    # ---------------- Per-frame helpers ----------------

    def _update_fps(self):
        self._fps_frame_counter += 1
        now = time.time()
        elapsed = now - self._fps_last_time
        if elapsed >= 0.5:
            self.fps = self._fps_frame_counter / elapsed
            self._fps_frame_counter = 0
            self._fps_last_time = now

    def get_window_size(self):
        try:
            rect = cv2.getWindowImageRect(WINDOW_NAME)
            w, h = int(rect[2]), int(rect[3])
            if w > 10 and h > 10:
                self.window_w, self.window_h = w, h
                return w, h
        except Exception:
            pass
        return self.window_w, self.window_h

    # ---------------- UI ----------------

    def _draw_overlay(self, img, gesture):
        h, w = img.shape[:2]

        # Recording indicator (top-left, only while actively recording)
        if self.recording:
            cv2.circle(img, (34, 34), 6, (0, 0, 255), -1)
            cv2.putText(img, "REC", (48, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # Top-center: AUTO GESTURE / MANUAL BLUR mode
        mode_text = "MANUAL BLUR" if self.manual_mode else "AUTO GESTURE"
        mw = 170
        draw_rounded_panel(img, (w - mw) // 2, 20, mw, 36)
        cv2.putText(img, mode_text, ((w - mw) // 2 + 14, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom-center: single combined panel (gesture label + blur status),
        # sized and margined to always fit inside the frame regardless of
        # window/fullscreen height.
        label = GESTURE_LABELS.get(gesture, "NORMAL")
        gesture_color = (255, 255, 255) if gesture == "peace" else (190, 190, 190)

        blur_on = self.blur_ctrl.current_strength > 0.05
        blur_text = "BLUR ON" if blur_on else "BLUR OFF"
        blur_color = (0, 220, 120) if blur_on else (140, 140, 140)

        panel_w = min(360, w - 40)
        panel_h = 70
        px = (w - panel_w) // 2
        py = h - panel_h - max(20, int(h * 0.03))
        py = max(py, 10)  # never let the panel go off the top either

        draw_rounded_panel(img, px, py, panel_w, panel_h)

        divider_x = px + int(panel_w * 0.55)
        cv2.line(img, (divider_x, py + 14), (divider_x, py + panel_h - 14), (80, 80, 80), 1)

        cv2.putText(img, "GESTURE", (px + 18, py + 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, label, (px + 18, py + 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, gesture_color, 2, cv2.LINE_AA)

        cv2.putText(img, blur_text, (divider_x + 16, py + 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, blur_color, 2, cv2.LINE_AA)

    # ---------------- Key handling ----------------

    def _handle_key(self, key, screenshot_source):
        low = key & 0xFF

        if low in (ord('q'), ord('Q')):
            return False

        elif key == 27:  # ESC -> exit fullscreen only
            if self.is_fullscreen:
                self.toggle_fullscreen()

        elif low in (ord('f'), ord('F')) or key in F11_KEYCODES:
            # 'F' is a guaranteed fullscreen toggle; F11 is best-effort
            # since its raw key code varies across OpenCV builds/OS.
            self.toggle_fullscreen()

        elif low in (ord('h'), ord('H')):
            self.show_ui = not self.show_ui

        elif low in (ord('b'), ord('B')):
            self._cycle_manual_blur()

        elif low in (ord('m'), ord('M')):
            self.mirror = not self.mirror

        elif low == ord('1'):
            self.blur_ctrl.set_mode("light")
        elif low == ord('2'):
            self.blur_ctrl.set_mode("medium")
        elif low == ord('3'):
            self.blur_ctrl.set_mode("heavy")

        elif low in (ord('r'), ord('R')):
            self._toggle_recording()

        elif low in (ord('c'), ord('C')):
            self._reset_settings()

        elif key == 32:  # SPACE
            self._take_screenshot(screenshot_source)

        elif low == ord('['):
            self._adjust_sensitivity(-0.01)
        elif low == ord(']'):
            self._adjust_sensitivity(0.01)

        return True

    def _cycle_manual_blur(self):
        """
        B cycles: AUTO GESTURE -> MANUAL (blur ON) -> MANUAL (blur OFF) -> AUTO GESTURE
        Gesture detection keeps running the whole time; only the blur
        *output* is overridden while in manual mode.
        """
        if not self.manual_mode:
            self.manual_mode = True
            self.manual_blur_on = True
        elif self.manual_blur_on:
            self.manual_blur_on = False
        else:
            self.manual_mode = False
            self.manual_blur_on = False

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, self.window_w, self.window_h)

    def _reset_settings(self):
        global FINGER_EXTENSION_THRESHOLD, THUMB_EXTENSION_THRESHOLD
        self.mirror = True
        self.manual_mode = False
        self.manual_blur_on = False
        self.blur_ctrl.set_mode(DEFAULT_BLUR_MODE)
        FINGER_EXTENSION_THRESHOLD = 0.12
        THUMB_EXTENSION_THRESHOLD = 0.06
        print("Settings reset to default.")

    def _adjust_sensitivity(self, delta):
        global FINGER_EXTENSION_THRESHOLD, THUMB_EXTENSION_THRESHOLD
        FINGER_EXTENSION_THRESHOLD = round(min(0.30, max(0.02, FINGER_EXTENSION_THRESHOLD + delta)), 3)
        THUMB_EXTENSION_THRESHOLD = round(min(0.30, max(0.02, THUMB_EXTENSION_THRESHOLD + delta)), 3)
        print(f"Gesture sensitivity -> finger={FINGER_EXTENSION_THRESHOLD}, thumb={THUMB_EXTENSION_THRESHOLD}")

    # ---------------- Screenshot / Recording ----------------

    def _take_screenshot(self, frame):
        filename = datetime.now().strftime("photo_%Y%m%d_%H%M%S.jpg")
        path = os.path.join(SCREENSHOT_DIR, filename)
        cv2.imwrite(path, frame)
        print(f"Screenshot saved: {path}")

    def _toggle_recording(self):
        if not self.recording:
            filename = datetime.now().strftime("video_%Y%m%d_%H%M%S.mp4")
            path = os.path.join(RECORDING_DIR, filename)

            frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_for_writer = 30.0

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, fps_for_writer, (frame_w, frame_h))

            if not writer.isOpened():
                # Fallback codec for systems missing mp4v support.
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(path, fourcc, fps_for_writer, (frame_w, frame_h))

            if not writer.isOpened():
                print("ERROR: Could not start recording (no compatible codec found).")
                return

            self.video_writer = writer
            self.recording = True
            print(f"Recording started: {path}")
        else:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            print("Recording stopped.")

    # ---------------- Cleanup ----------------

    def _cleanup(self):
        if self.video_writer is not None:
            self.video_writer.release()
        if self.cap is not None:
            self.cap.release()
        try:
            self.landmarker.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = GestureBlurCameraApp()
    app.run()