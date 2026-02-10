"""
AutoTracker.py -- Auto-Tracking Deepfake Screen Scanner
=======================================================
Hunts for faces on the full screen, auto-locks a sniper-scope overlay,
and runs the DeepfakeGRU temporal model on the captured region.

Architecture:
  1. GlobalScannerThread  -- Full-screen face detection (MTCNN, downscaled)
  2. ModelThread           -- EfficientNet feature extraction + GRU inference
  3. SniperOverlay         -- Transparent PyQt6 window with auto-aim + crosshair

Color States:
  GREY   = Searching for face
  YELLOW = Locked on face, filling frame buffer
  GREEN  = Model says REAL
  RED    = Model says FAKE
"""

import sys
import os
import math
import warnings
import traceback

warnings.filterwarnings("ignore")

# -- Import torch BEFORE PyQt6 to avoid Windows DLL conflicts --
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torchvision import transforms

import cv2
from mss import mss

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen, QFont

# ===================================================================
#  Dependency checks
# ===================================================================
_MODELS_AVAILABLE = False

try:
    import timm
    from facenet_pytorch import MTCNN
    _MODELS_AVAILABLE = True
except ImportError:
    print("[WARN] timm / facenet_pytorch not installed -- models disabled")


# ===================================================================
#  DeepfakeGRU (self-contained copy from realtime_inference.py)
# ===================================================================
class DeepfakeGRU(nn.Module):
    """GRU temporal classifier over EfficientNet feature sequences."""

    def __init__(self, input_dim=1280, hidden_dim=256, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])


# ===================================================================
#  1. GlobalScannerThread -- Full-Screen Face Hunting
# ===================================================================
class GlobalScannerThread(QThread):
    """Scans the entire screen for faces at ~5 FPS (downscaled).
    Emits face coordinates when found, or no_face when lost."""

    face_found = pyqtSignal(int, int, int, int)   # x, y, w, h in screen coords
    no_face    = pyqtSignal()

    SCAN_INTERVAL_MS = 200      # ~5 FPS scanning
    SCALE_FACTOR     = 0.35     # downscale for speed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.device = torch.device("cpu")   # face detection always on CPU for speed
        self.mtcnn = None

        if not _MODELS_AVAILABLE:
            return

        try:
            self.mtcnn = MTCNN(
                image_size=160, margin=30,
                keep_all=True, device=self.device,
                thresholds=[0.6, 0.7, 0.8],
            )
            print("[OK] GlobalScanner MTCNN ready")
        except Exception as e:
            print(f"[ERR] GlobalScanner MTCNN failed: {e}")

    def run(self):
        if self.mtcnn is None:
            return

        with mss() as sct:
            monitor = sct.monitors[1]   # primary monitor
            mon_left = monitor["left"]
            mon_top  = monitor["top"]

            while self.running:
                try:
                    screenshot = sct.grab(monitor)
                    img = np.array(screenshot)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

                    h, w = img_rgb.shape[:2]
                    sw = int(w * self.SCALE_FACTOR)
                    sh = int(h * self.SCALE_FACTOR)
                    small = cv2.resize(img_rgb, (sw, sh))

                    boxes, probs = self.mtcnn.detect(small)

                    if boxes is not None and len(boxes) > 0:
                        # Pick the highest-confidence face
                        best_idx = int(np.argmax(probs))
                        x1, y1, x2, y2 = boxes[best_idx]

                        # Scale back to screen coords
                        fx = int(x1 / self.SCALE_FACTOR) + mon_left
                        fy = int(y1 / self.SCALE_FACTOR) + mon_top
                        fw = int((x2 - x1) / self.SCALE_FACTOR)
                        fh = int((y2 - y1) / self.SCALE_FACTOR)

                        # Add padding
                        pad = int(min(fw, fh) * 0.35)
                        fx = max(0, fx - pad)
                        fy = max(0, fy - pad)
                        fw += pad * 2
                        fh += pad * 2

                        self.face_found.emit(fx, fy, fw, fh)
                    else:
                        self.no_face.emit()
                except Exception:
                    pass

                self.msleep(self.SCAN_INTERVAL_MS)

    def stop(self):
        self.running = False
        self.wait()


# ===================================================================
#  2. ModelThread -- EfficientNet + GRU Inference on Locked Region
# ===================================================================
class ModelThread(QThread):
    """When the overlay is locked on a face, captures that region,
    extracts EfficientNet features, buffers them, and runs GRU."""

    prediction   = pyqtSignal(float)    # 0=real ... 1=fake
    buffer_status = pyqtSignal(int, int) # current, total
    status_msg   = pyqtSignal(str)

    SEQUENCE_LENGTH = 10
    CAPTURE_INTERVAL_MS = 150   # ~6-7 FPS capture from locked region

    def __init__(self, capture_rect_fn, parent=None):
        super().__init__(parent)
        self.running = True
        self.locked = False   # controlled by main thread
        self.capture_rect_fn = capture_rect_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn = None
        self.rnn_model = None
        self.mtcnn_crop = None

        if not _MODELS_AVAILABLE:
            return

        # -- EfficientNet feature extractor --
        try:
            self.cnn = timm.create_model(
                "tf_efficientnetv2_s", pretrained=False, num_classes=0,
            )
            cnn_path = os.path.join(
                os.path.dirname(__file__),
                "models", "weights", "VIDEO", "best_ffpp_efficientnet.pth",
            )
            if os.path.exists(cnn_path):
                self.cnn.load_state_dict(
                    torch.load(cnn_path, map_location=self.device), strict=False,
                )
                print(f"[OK] EfficientNet loaded: {cnn_path}")
            else:
                print(f"[WARN] CNN weights missing: {cnn_path}")
            self.cnn = self.cnn.to(self.device)
            self.cnn.eval()
        except Exception as e:
            print(f"[ERR] EfficientNet init: {e}")
            self.cnn = None

        # -- GRU temporal model --
        try:
            self.rnn_model = DeepfakeGRU(input_dim=1280, hidden_dim=256, num_layers=1)
            rnn_path = os.path.join(
                os.path.dirname(__file__),
                "models", "weights", "VIDEO", "best_rnn.pt",
            )
            if os.path.exists(rnn_path):
                self.rnn_model.load_state_dict(
                    torch.load(rnn_path, map_location=self.device),
                )
                print(f"[OK] GRU loaded: {rnn_path}")
            else:
                print(f"[WARN] GRU weights missing: {rnn_path}")
            self.rnn_model = self.rnn_model.to(self.device)
            self.rnn_model.eval()
        except Exception as e:
            print(f"[ERR] GRU init: {e}")
            self.rnn_model = None

        # -- MTCNN for precise face crop within locked region --
        try:
            self.mtcnn_crop = MTCNN(
                image_size=224, margin=20,
                keep_all=False, device=self.device,
            )
        except Exception:
            self.mtcnn_crop = None

        self.feature_buffer = deque(maxlen=self.SEQUENCE_LENGTH)

    @torch.no_grad()
    def _extract_feature(self, frame_bgr):
        """BGR frame -> 1280-dim feature or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.mtcnn_crop is not None:
            face = self.mtcnn_crop(rgb)
            if face is None:
                return None
        else:
            # Fallback: resize the whole region
            from torchvision import transforms as T
            face = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])(rgb)

        face = face.unsqueeze(0).to(self.device)   # (1, 3, 224, 224)
        feat = self.cnn(face)                       # (1, 1280)
        return feat.squeeze(0)                      # (1280,)

    def run(self):
        if self.cnn is None or self.rnn_model is None:
            self.status_msg.emit("MODEL: OFF (weights not loaded)")
            return

        with mss() as sct:
            while self.running:
                if not self.locked:
                    # Not locked -- clear buffer and wait
                    if len(self.feature_buffer) > 0:
                        self.feature_buffer.clear()
                    self.msleep(100)
                    continue

                try:
                    rect = self.capture_rect_fn()
                    screenshot = sct.grab(rect)
                    frame = np.array(screenshot)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    feat = self._extract_feature(frame_bgr)
                    if feat is not None:
                        self.feature_buffer.append(feat)

                    buf_len = len(self.feature_buffer)
                    self.buffer_status.emit(buf_len, self.SEQUENCE_LENGTH)

                    if buf_len >= self.SEQUENCE_LENGTH:
                        seq = torch.stack(list(self.feature_buffer))
                        seq = seq.unsqueeze(0).to(self.device)  # (1, T, 1280)

                        with torch.no_grad():
                            logit = self.rnn_model(seq)
                            prob = torch.sigmoid(logit).item()

                        self.prediction.emit(prob)

                except Exception:
                    pass

                self.msleep(self.CAPTURE_INTERVAL_MS)

    def stop(self):
        self.running = False
        self.wait()


# ===================================================================
#  3. SniperOverlay -- Auto-Aiming Transparent UI
# ===================================================================
class SniperOverlay(QMainWindow):
    """Transparent sniper-scope overlay that auto-tracks faces on screen
    and displays deepfake classification with color-coded feedback."""

    # --- States ---
    STATE_SEARCHING = "SEARCHING"
    STATE_ACQUIRING = "ACQUIRING"
    STATE_LOCKED    = "LOCKED"

    # --- Thresholds ---
    FAKE_THRESHOLD = 0.5
    LERP_SPEED     = 0.25       # lower = smoother, higher = snappier

    def __init__(self):
        super().__init__()

        # -- geometry --
        self.window_size   = 320
        self.border_width  = 4
        self.corner_length = 35

        # -- tracking state --
        self.current_x = 0
        self.current_y = 0
        self.target_x  = 0
        self.target_y  = 0
        self.state = self.STATE_SEARCHING
        self.manual_mode = False
        self.dragging = False
        self.drag_offset = QPoint()
        self.spin_angle = 0

        # -- scores --
        self.latest_prob = -1.0     # -1 = no prediction yet
        self.buffer_fill = 0
        self.buffer_max  = 20

        # -- colors --
        self.COLOR_SEARCHING = QColor(120, 130, 140)       # grey
        self.COLOR_ACQUIRING = QColor(255, 200, 0)         # yellow
        self.COLOR_REAL      = QColor(0, 255, 100)         # green
        self.COLOR_FAKE      = QColor(255, 40, 60)         # red
        self.current_color   = self.COLOR_SEARCHING

        self._init_ui()
        self._init_threads()
        self._init_timers()

    # ---------------------------------------------------------------
    #  UI
    # ---------------------------------------------------------------
    def _init_ui(self):
        self.setWindowTitle("AutoTracker")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        screen = QApplication.primaryScreen().geometry()
        self.current_x = (screen.width()  - self.window_size) // 2
        self.current_y = (screen.height() - self.window_size) // 2
        self.target_x = self.current_x
        self.target_y = self.current_y
        self.setGeometry(self.current_x, self.current_y,
                         self.window_size, self.window_size)

        # Labels
        lbl_style = (
            "color: white; background-color: rgba(0,0,0,180);"
            "border-radius: 4px; padding: 2px 8px; font-weight: bold;"
            "font-size: 12px;"
        )

        self.status_label = QLabel("SCANNING...", self)
        self.status_label.setStyleSheet(lbl_style)
        self.status_label.setFixedWidth(180)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.detail_label = QLabel("", self)
        self.detail_label.setStyleSheet(
            "color: rgba(255,255,255,180); background-color: rgba(0,0,0,140);"
            "border-radius: 3px; padding: 1px 6px; font-size: 10px;"
        )
        self.detail_label.setFixedWidth(160)
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.mode_label = QLabel("AUTO", self)
        self.mode_label.setStyleSheet(
            "color: #00ff64; background-color: rgba(0,0,0,180);"
            "border-radius: 3px; padding: 2px 6px; font-weight: bold;"
            "font-size: 10px;"
        )

        self._reposition_labels()

    def _reposition_labels(self):
        w = self.window_size
        self.status_label.move((w - 180) // 2, w - 50)
        self.detail_label.move((w - 160) // 2, w - 28)
        self.mode_label.move(8, 8)

    # ---------------------------------------------------------------
    #  Threads
    # ---------------------------------------------------------------
    def _init_threads(self):
        # Face hunter
        self.scanner = GlobalScannerThread()
        self.scanner.face_found.connect(self._on_face_found)
        self.scanner.no_face.connect(self._on_no_face)
        self.scanner.start()

        # Model inference
        self.model_thread = ModelThread(
            capture_rect_fn=self._get_capture_rect,
        )
        self.model_thread.prediction.connect(self._on_prediction)
        self.model_thread.buffer_status.connect(self._on_buffer_status)
        self.model_thread.status_msg.connect(self._on_model_status)
        self.model_thread.start()

    def _init_timers(self):
        # Smooth position lerp at 60 FPS
        self.move_timer = QTimer()
        self.move_timer.timeout.connect(self._update_position)
        self.move_timer.start(16)

        # Spin animation at 30 FPS
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._tick_animation)
        self.anim_timer.start(33)

        # Face-lost timeout (if no face for 1.5s, go back to searching)
        self.face_timeout = QTimer()
        self.face_timeout.setSingleShot(True)
        self.face_timeout.timeout.connect(self._on_face_lost_timeout)

    def _get_capture_rect(self):
        return {
            "left":   self.x(),
            "top":    self.y(),
            "width":  self.window_size,
            "height": self.window_size,
        }

    # ---------------------------------------------------------------
    #  Slots -- Face Tracking
    # ---------------------------------------------------------------
    def _on_face_found(self, x, y, w, h):
        if self.manual_mode:
            return

        # Center the window on the face
        self.target_x = x + (w - self.window_size) // 2
        self.target_y = y + (h - self.window_size) // 2

        # Restart the face-lost timeout
        self.face_timeout.start(1500)

        if self.state == self.STATE_SEARCHING:
            self.state = self.STATE_ACQUIRING
            self.model_thread.locked = True
            self.latest_prob = -1.0
            self.current_color = self.COLOR_ACQUIRING
            self.status_label.setText("ACQUIRING...")
            self.update()

    def _on_no_face(self):
        # Don't immediately lose lock -- wait for timeout
        pass

    def _on_face_lost_timeout(self):
        if self.manual_mode:
            return
        self.state = self.STATE_SEARCHING
        self.model_thread.locked = False
        self.latest_prob = -1.0
        self.buffer_fill = 0
        self.current_color = self.COLOR_SEARCHING
        self.status_label.setText("SCANNING...")
        self.detail_label.setText("")
        self.update()

    # ---------------------------------------------------------------
    #  Slots -- Model Inference
    # ---------------------------------------------------------------
    def _on_prediction(self, prob):
        self.latest_prob = prob
        self.state = self.STATE_LOCKED

        if prob > self.FAKE_THRESHOLD:
            self.current_color = self.COLOR_FAKE
            pct = int(prob * 100)
            self.status_label.setText(f"FAKE {pct}%")
        else:
            self.current_color = self.COLOR_REAL
            pct = int((1 - prob) * 100)
            self.status_label.setText(f"REAL {pct}%")

        self.detail_label.setText(f"prob: {prob:.3f}")
        self.update()

    def _on_buffer_status(self, current, total):
        self.buffer_fill = current
        self.buffer_max = total
        if self.state == self.STATE_ACQUIRING:
            self.detail_label.setText(f"buffer: {current}/{total}")

    def _on_model_status(self, msg):
        self.detail_label.setText(msg)

    # ---------------------------------------------------------------
    #  Animation & Movement
    # ---------------------------------------------------------------
    def _update_position(self):
        if self.manual_mode or self.dragging:
            return

        # Smooth lerp
        s = self.LERP_SPEED
        self.current_x = int(self.current_x * (1 - s) + self.target_x * s)
        self.current_y = int(self.current_y * (1 - s) + self.target_y * s)
        self.move(self.current_x, self.current_y)

    def _tick_animation(self):
        self.spin_angle = (self.spin_angle + 2) % 360
        self.update()

    # ===============================================================
    #  PAINT -- Sniper Scope + Crosshair + Buffer Arc
    # ===============================================================
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.window_size
        cx, cy = w // 2, w // 2
        bw = self.border_width
        cl = self.corner_length
        color = self.current_color

        # -- 1. Outer ring (subtle glow) --
        ring_pen = QPen(QColor(color.red(), color.green(), color.blue(), 50), 2)
        p.setPen(ring_pen)
        p.drawEllipse(8, 8, w - 16, w - 16)

        # -- 2. Spinning tick marks --
        tick_pen = QPen(QColor(color.red(), color.green(), color.blue(), 130), 2)
        p.setPen(tick_pen)
        radius = (w - 16) / 2
        for i in range(12):
            angle_rad = math.radians(self.spin_angle + i * 30)
            x1 = cx + (radius - 7) * math.cos(angle_rad)
            y1 = cy + (radius - 7) * math.sin(angle_rad)
            x2 = cx + (radius - 1) * math.cos(angle_rad)
            y2 = cy + (radius - 1) * math.sin(angle_rad)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # -- 3. Corner brackets --
        corner_pen = QPen(color, bw)
        p.setPen(corner_pen)
        b = bw // 2

        p.drawLine(b, b, b + cl, b)
        p.drawLine(b, b, b, b + cl)

        p.drawLine(w - b - cl, b, w - b, b)
        p.drawLine(w - b, b, w - b, b + cl)

        p.drawLine(b, w - b, b + cl, w - b)
        p.drawLine(b, w - b - cl, b, w - b)

        p.drawLine(w - b - cl, w - b, w - b, w - b)
        p.drawLine(w - b, w - b - cl, w - b, w - b)

        # -- 4. Crosshair with targeting gap --
        cross_pen = QPen(QColor(color.red(), color.green(), color.blue(), 180), 1)
        p.setPen(cross_pen)
        gap = 14
        arm = 30

        p.drawLine(cx - arm, cy, cx - gap, cy)
        p.drawLine(cx + gap, cy, cx + arm, cy)
        p.drawLine(cx, cy - arm, cx, cy - gap)
        p.drawLine(cx, cy + gap, cx, cy + arm)

        # Center dot
        p.setPen(QPen(color, 3))
        p.drawPoint(cx, cy)

        # -- 5. Buffer progress arc (shows how full the frame buffer is) --
        if self.buffer_max > 0 and self.state in (self.STATE_ACQUIRING, self.STATE_LOCKED):
            fill_ratio = min(self.buffer_fill / self.buffer_max, 1.0)
            arc_pen = QPen(QColor(color.red(), color.green(), color.blue(), 160), 3)
            p.setPen(arc_pen)
            margin = 22
            span = int(fill_ratio * 360 * 16)
            p.drawArc(
                margin, margin,
                w - margin * 2, w - margin * 2,
                90 * 16,    # 12 o'clock
                -span,      # clockwise
            )

        # -- 6. Confidence arc (only when we have a prediction) --
        if self.latest_prob >= 0:
            conf_pen = QPen(color, 4)
            p.setPen(conf_pen)
            margin2 = 30
            conf_span = int(abs(self.latest_prob - 0.5) * 2 * 360 * 16)
            p.drawArc(
                margin2, margin2,
                w - margin2 * 2, w - margin2 * 2,
                90 * 16,
                -conf_span,
            )

        p.end()

    # ===============================================================
    #  INPUT -- Keyboard + Mouse
    # ===============================================================
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_M:
            self.manual_mode = not self.manual_mode
            if self.manual_mode:
                self.mode_label.setText("MANUAL")
                self.mode_label.setStyleSheet(
                    "color: #ffaa00; background-color: rgba(0,0,0,180);"
                    "border-radius: 3px; padding: 2px 6px; font-weight: bold;"
                    "font-size: 10px;"
                )
                # Lock model when entering manual mode
                self.model_thread.locked = True
                if self.state == self.STATE_SEARCHING:
                    self.state = self.STATE_ACQUIRING
                    self.current_color = self.COLOR_ACQUIRING
                    self.status_label.setText("ACQUIRING...")
                    self.update()
            else:
                self.mode_label.setText("AUTO")
                self.mode_label.setStyleSheet(
                    "color: #00ff64; background-color: rgba(0,0,0,180);"
                    "border-radius: 3px; padding: 2px 6px; font-weight: bold;"
                    "font-size: 10px;"
                )
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.window_size = min(600, self.window_size + 20)
            self.resize(self.window_size, self.window_size)
            self._reposition_labels()
        elif key == Qt.Key.Key_Minus:
            self.window_size = max(150, self.window_size - 20)
            self.resize(self.window_size, self.window_size)
            self._reposition_labels()
        elif key == Qt.Key.Key_R:
            # Reset buffer
            self.model_thread.feature_buffer.clear()
            self.latest_prob = -1.0
            self.buffer_fill = 0
            if self.state == self.STATE_LOCKED:
                self.state = self.STATE_ACQUIRING
                self.current_color = self.COLOR_ACQUIRING
                self.status_label.setText("ACQUIRING...")
                self.detail_label.setText("")
                self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.manual_mode:
            self.dragging = True
            self.drag_offset = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging and self.manual_mode:
            self.move(event.globalPosition().toPoint() - self.drag_offset)

    def mouseReleaseEvent(self, event):
        self.dragging = False

    # ===============================================================
    #  CLEANUP
    # ===============================================================
    def closeEvent(self, event):
        print("\nShutting down AutoTracker...")
        self.move_timer.stop()
        self.anim_timer.stop()
        self.face_timeout.stop()
        self.scanner.stop()
        self.model_thread.stop()
        event.accept()


# ===================================================================
#  MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  AutoTracker -- Screen Deepfake Scanner")
    print("=" * 60)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device  : {dev}")
    print(f"  Models  : {'ENABLED' if _MODELS_AVAILABLE else 'DISABLED'}")
    print("=" * 60)
    print("  Keys:")
    print("    M     = Toggle manual/auto mode")
    print("    R     = Reset frame buffer")
    print("    +/-   = Resize scope window")
    print("    Esc   = Quit")
    print("=" * 60)

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    overlay = SniperOverlay()
    overlay.show()

    sys.exit(app.exec())
