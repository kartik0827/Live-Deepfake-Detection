"""
FusionScanner.py -- Auto-Tracking Deepfake Fusion Scanner
==========================================================
Hunts for faces on screen, auto-locks, and runs TWO pipelines:
  Visual : EfficientNet + GRU temporal model on captured frames
  Audio  : AASIST anti-spoofing on live microphone input
Scores are fused in real-time on a sniper-scope overlay.

Threads:
  1. GlobalScannerThread  -- Full-screen face hunting (MTCNN, downscaled)
  2. ModelThread           -- EfficientNet features + GRU inference
  3. AudioScannerThread    -- PyAudio microphone + AASIST inference
  4. Main UI              -- Lerp tracking + painting + fusion

Color States:
  GREY   = Searching for face
  YELLOW = Acquired, filling buffer
  GREEN  = Fusion says REAL
  RED    = Fusion says FAKE
"""

import sys
import os
import math
import warnings
import traceback

warnings.filterwarnings("ignore")

# -- torch before PyQt6 (Windows DLL fix) --
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
#  Dependency checks (graceful fallback)
# ===================================================================
_VISUAL_AVAILABLE = False
_AUDIO_AVAILABLE  = False

try:
    import timm
    from facenet_pytorch import MTCNN
    _VISUAL_AVAILABLE = True
except ImportError:
    print("[WARN] timm / facenet_pytorch not installed -- visual pipeline disabled")

try:
    from AASISTMODEL import LiveAASISTDetector
    _AUDIO_AVAILABLE = True
except ImportError:
    print("[WARN] AASISTMODEL not found -- audio pipeline disabled")

try:
    import pyaudio
except ImportError:
    pyaudio = None
    print("[WARN] pyaudio not installed -- audio capture disabled")


# ===================================================================
#  DeepfakeGRU (from realtime_inference.py)
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
    """Scans entire screen for faces at ~5 FPS (downscaled).
    Emits coordinates when a face is found."""

    face_found = pyqtSignal(int, int, int, int)   # x, y, w, h
    no_face    = pyqtSignal()

    SCAN_INTERVAL_MS = 200
    SCALE_FACTOR     = 0.35

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.device = torch.device("cpu")
        self.mtcnn = None

        if not _VISUAL_AVAILABLE:
            return

        try:
            self.mtcnn = MTCNN(
                image_size=160, margin=30,
                keep_all=True, device=self.device,
                thresholds=[0.6, 0.7, 0.8],
            )
            print("[OK] GlobalScanner MTCNN ready")
        except Exception as e:
            print(f"[ERR] GlobalScanner MTCNN: {e}")

    def run(self):
        if self.mtcnn is None:
            return

        with mss() as sct:
            monitor = sct.monitors[1]
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
                        best = int(np.argmax(probs))
                        x1, y1, x2, y2 = boxes[best]

                        fx = int(x1 / self.SCALE_FACTOR) + mon_left
                        fy = int(y1 / self.SCALE_FACTOR) + mon_top
                        fw = int((x2 - x1) / self.SCALE_FACTOR)
                        fh = int((y2 - y1) / self.SCALE_FACTOR)

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
#  2. ModelThread -- EfficientNet + GRU on Locked Region
# ===================================================================
class ModelThread(QThread):
    """Captures the locked region, extracts features, runs GRU."""

    prediction    = pyqtSignal(float)     # 0=real ... 1=fake
    buffer_status = pyqtSignal(int, int)  # current, total
    status_msg    = pyqtSignal(str)

    SEQUENCE_LENGTH     = 10
    CAPTURE_INTERVAL_MS = 150

    def __init__(self, capture_rect_fn, parent=None):
        super().__init__(parent)
        self.running = True
        self.locked = False
        self.capture_rect_fn = capture_rect_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn = None
        self.rnn_model = None
        self.mtcnn_crop = None

        if not _VISUAL_AVAILABLE:
            return

        # -- EfficientNet --
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
            print(f"[ERR] EfficientNet: {e}")
            self.cnn = None

        # -- GRU --
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
            print(f"[ERR] GRU: {e}")
            self.rnn_model = None

        # -- MTCNN for precise crop --
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
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.mtcnn_crop is not None:
            face = self.mtcnn_crop(rgb)
            if face is None:
                return None
        else:
            from torchvision import transforms as T
            face = T.Compose([
                T.ToPILImage(), T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])(rgb)

        face = face.unsqueeze(0).to(self.device)
        feat = self.cnn(face)
        return feat.squeeze(0)

    def run(self):
        if self.cnn is None or self.rnn_model is None:
            self.status_msg.emit("VIDEO: OFF (weights not loaded)")
            return

        with mss() as sct:
            while self.running:
                if not self.locked:
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
                        seq = seq.unsqueeze(0).to(self.device)

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
#  3. AudioScannerThread -- Microphone + AASIST
# ===================================================================
class AudioScannerThread(QThread):
    """Captures mic audio via PyAudio, buffers 4 seconds,
    runs AASIST inference every 2 seconds."""

    audio_score = pyqtSignal(float)    # 0=real ... 1=fake
    status_msg  = pyqtSignal(str)

    SAMPLE_RATE     = 16000
    BUFFER_DURATION = 4.0
    INFER_INTERVAL  = 2.0
    CHUNK_SIZE      = 1024

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.detector = None

        if not _AUDIO_AVAILABLE:
            return
        if pyaudio is None:
            return

        try:
            aasist_path = os.path.join(
                os.path.dirname(__file__),
                "models", "weights", "AASIST", "AASIST.pth",
            )
            self.detector = LiveAASISTDetector(model_path=aasist_path)
            print(f"[OK] AASIST loaded: {aasist_path}")
        except Exception as e:
            print(f"[ERR] AASIST: {e}")
            traceback.print_exc()
            self.detector = None

    def run(self):
        if self.detector is None or pyaudio is None:
            self.status_msg.emit("AUDIO: OFF")
            return

        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
            )

            buffer_size = int(self.SAMPLE_RATE * self.BUFFER_DURATION)
            audio_buffer = np.zeros(buffer_size, dtype=np.float32)
            samples_since_infer = 0
            infer_samples = int(self.SAMPLE_RATE * self.INFER_INTERVAL)

            self.status_msg.emit("AUDIO: listening...")

            while self.running:
                try:
                    raw = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    chunk = np.frombuffer(raw, dtype=np.float32)

                    audio_buffer = np.roll(audio_buffer, -len(chunk))
                    audio_buffer[-len(chunk):] = chunk
                    samples_since_infer += len(chunk)

                    if samples_since_infer >= infer_samples:
                        samples_since_infer = 0
                        prob = self.detector.detect(audio_buffer.copy())
                        self.audio_score.emit(prob)
                except Exception:
                    pass

        except Exception as e:
            self.status_msg.emit(f"AUDIO: error ({e})")
            print(f"[ERR] Audio stream: {e}")
            traceback.print_exc()
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.wait()


# ===================================================================
#  4. FusionOverlay -- Auto-Tracking + Fused Score Display
# ===================================================================
class FusionOverlay(QMainWindow):
    """Auto-tracking sniper-scope that fuses visual + audio scores."""

    # States
    STATE_SEARCHING = "SEARCHING"
    STATE_ACQUIRING = "ACQUIRING"
    STATE_LOCKED    = "LOCKED"

    # Weights
    VISUAL_WEIGHT = 0.7
    AUDIO_WEIGHT  = 0.3
    THRESHOLD     = 0.5
    LERP_SPEED    = 0.25

    def __init__(self):
        super().__init__()

        # -- geometry --
        self.window_size   = 340
        self.border_width  = 4
        self.corner_length = 35

        # -- tracking --
        self.current_x = 0
        self.current_y = 0
        self.target_x  = 0
        self.target_y  = 0
        self.state = self.STATE_SEARCHING
        self.manual_mode = False
        self.dragging    = False
        self.drag_offset = QPoint()
        self.spin_angle  = 0

        # -- scores --
        self.latest_visual = -1.0   # -1 = no prediction yet
        self.latest_audio  = -1.0
        self.combined      = -1.0
        self.buffer_fill   = 0
        self.buffer_max    = 10

        # -- colors --
        self.COLOR_SEARCHING = QColor(120, 130, 140)   # grey
        self.COLOR_ACQUIRING = QColor(255, 200, 0)     # yellow
        self.COLOR_REAL      = QColor(0, 255, 100)     # green
        self.COLOR_FAKE      = QColor(255, 40, 60)     # red
        self.current_color   = self.COLOR_SEARCHING

        self._init_ui()
        self._init_threads()
        self._init_timers()

    # ---------------------------------------------------------------
    #  UI
    # ---------------------------------------------------------------
    def _init_ui(self):
        self.setWindowTitle("FusionScanner")
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

        lbl_style = (
            "color: white; background-color: rgba(0,0,0,180);"
            "border-radius: 4px; padding: 2px 8px; font-weight: bold;"
            "font-size: 11px;"
        )

        self.video_label = QLabel("VIDEO: -", self)
        self.video_label.setStyleSheet(lbl_style)
        self.video_label.setFixedWidth(170)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.audio_lbl = QLabel("AUDIO: -", self)
        self.audio_lbl.setStyleSheet(lbl_style)
        self.audio_lbl.setFixedWidth(170)
        self.audio_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.combined_label = QLabel("FUSION: -", self)
        self.combined_label.setStyleSheet(
            "color: white; background-color: rgba(0,0,0,200);"
            "border-radius: 5px; padding: 3px 8px; font-weight: bold;"
            "font-size: 13px;"
        )
        self.combined_label.setFixedWidth(190)
        self.combined_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.mode_label = QLabel("AUTO", self)
        self.mode_label.setStyleSheet(
            "color: #00ff64; background-color: rgba(0,0,0,180);"
            "border-radius: 3px; padding: 2px 6px; font-weight: bold;"
            "font-size: 10px;"
        )

        self._reposition_labels()

    def _reposition_labels(self):
        w = self.window_size
        self.video_label.move((w - 170) // 2, w - 70)
        self.audio_lbl.move((w - 170) // 2, w - 48)
        self.combined_label.move((w - 190) // 2, w - 26)
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

        # Visual model
        self.model_thread = ModelThread(
            capture_rect_fn=self._get_capture_rect,
        )
        self.model_thread.prediction.connect(self._on_visual_score)
        self.model_thread.buffer_status.connect(self._on_buffer_status)
        self.model_thread.status_msg.connect(self._on_visual_status)
        self.model_thread.start()

        # Audio model
        self.audio_thread = AudioScannerThread()
        self.audio_thread.audio_score.connect(self._on_audio_score)
        self.audio_thread.status_msg.connect(self._on_audio_status)
        self.audio_thread.start()

    def _init_timers(self):
        self.move_timer = QTimer()
        self.move_timer.timeout.connect(self._update_position)
        self.move_timer.start(16)

        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._tick_animation)
        self.anim_timer.start(33)

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

        self.target_x = x + (w - self.window_size) // 2
        self.target_y = y + (h - self.window_size) // 2
        self.face_timeout.start(1500)

        if self.state == self.STATE_SEARCHING:
            self.state = self.STATE_ACQUIRING
            self.model_thread.locked = True
            self.latest_visual = -1.0
            self.current_color = self.COLOR_ACQUIRING
            self.video_label.setText("VIDEO: acquiring...")
            self.combined_label.setText("FUSION: -")
            self.update()

    def _on_no_face(self):
        pass  # wait for timeout

    def _on_face_lost_timeout(self):
        if self.manual_mode:
            return
        self.state = self.STATE_SEARCHING
        self.model_thread.locked = False
        self.latest_visual = -1.0
        self.combined = -1.0
        self.buffer_fill = 0
        self.current_color = self.COLOR_SEARCHING
        self.video_label.setText("VIDEO: -")
        self.combined_label.setText("FUSION: -")
        self.update()

    # ---------------------------------------------------------------
    #  Slots -- Visual Score
    # ---------------------------------------------------------------
    def _on_visual_score(self, prob):
        self.latest_visual = prob
        self.state = self.STATE_LOCKED

        v_label = "FAKE" if prob > self.THRESHOLD else "REAL"
        pct = int(prob * 100) if prob > self.THRESHOLD else int((1 - prob) * 100)
        self.video_label.setText(f"VIDEO: {v_label} {pct}%")
        self._recalculate()

    def _on_buffer_status(self, current, total):
        self.buffer_fill = current
        self.buffer_max = total

    def _on_visual_status(self, msg):
        self.video_label.setText(msg)

    # ---------------------------------------------------------------
    #  Slots -- Audio Score
    # ---------------------------------------------------------------
    def _on_audio_score(self, prob):
        self.latest_audio = prob
        a_label = "FAKE" if prob > self.THRESHOLD else "REAL"
        pct = int(prob * 100) if prob > self.THRESHOLD else int((1 - prob) * 100)
        self.audio_lbl.setText(f"AUDIO: {a_label} {pct}%")
        self._recalculate()

    def _on_audio_status(self, msg):
        self.audio_lbl.setText(msg)

    # ---------------------------------------------------------------
    #  Fusion
    # ---------------------------------------------------------------
    def _recalculate(self):
        v = self.latest_visual
        a = self.latest_audio

        # Both available -> weighted fusion
        if v >= 0 and a >= 0:
            self.combined = v * self.VISUAL_WEIGHT + a * self.AUDIO_WEIGHT
        # Only visual
        elif v >= 0:
            self.combined = v
        # Only audio
        elif a >= 0:
            self.combined = a
        else:
            return

        if self.combined > self.THRESHOLD:
            self.current_color = self.COLOR_FAKE
            pct = int(self.combined * 100)
            self.combined_label.setText(f"FUSION: FAKE {pct}%")
        else:
            self.current_color = self.COLOR_REAL
            pct = int((1 - self.combined) * 100)
            self.combined_label.setText(f"FUSION: REAL {pct}%")

        self.update()

    # ---------------------------------------------------------------
    #  Animation & Movement
    # ---------------------------------------------------------------
    def _update_position(self):
        if self.manual_mode or self.dragging:
            return
        s = self.LERP_SPEED
        self.current_x = int(self.current_x * (1 - s) + self.target_x * s)
        self.current_y = int(self.current_y * (1 - s) + self.target_y * s)
        self.move(self.current_x, self.current_y)

    def _tick_animation(self):
        self.spin_angle = (self.spin_angle + 2) % 360
        self.update()

    # ===============================================================
    #  PAINT -- Sniper Scope + Crosshair + Arcs
    # ===============================================================
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.window_size
        cx, cy = w // 2, w // 2
        bw = self.border_width
        cl = self.corner_length
        color = self.current_color

        # 1. Outer ring
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 50), 2))
        p.drawEllipse(8, 8, w - 16, w - 16)

        # 2. Spinning ticks
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 130), 2))
        radius = (w - 16) / 2
        for i in range(12):
            a = math.radians(self.spin_angle + i * 30)
            x1 = cx + (radius - 7) * math.cos(a)
            y1 = cy + (radius - 7) * math.sin(a)
            x2 = cx + (radius - 1) * math.cos(a)
            y2 = cy + (radius - 1) * math.sin(a)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # 3. Corner brackets
        p.setPen(QPen(color, bw))
        b = bw // 2
        p.drawLine(b, b, b + cl, b)
        p.drawLine(b, b, b, b + cl)
        p.drawLine(w - b - cl, b, w - b, b)
        p.drawLine(w - b, b, w - b, b + cl)
        p.drawLine(b, w - b, b + cl, w - b)
        p.drawLine(b, w - b - cl, b, w - b)
        p.drawLine(w - b - cl, w - b, w - b, w - b)
        p.drawLine(w - b, w - b - cl, w - b, w - b)

        # 4. Crosshair
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 180), 1))
        gap, arm = 14, 30
        p.drawLine(cx - arm, cy, cx - gap, cy)
        p.drawLine(cx + gap, cy, cx + arm, cy)
        p.drawLine(cx, cy - arm, cx, cy - gap)
        p.drawLine(cx, cy + gap, cx, cy + arm)

        p.setPen(QPen(color, 3))
        p.drawPoint(cx, cy)

        # 5. Buffer arc (shows frame buffer fill)
        if self.buffer_max > 0 and self.state in (self.STATE_ACQUIRING, self.STATE_LOCKED):
            fill = min(self.buffer_fill / self.buffer_max, 1.0)
            p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 140), 3))
            m = 22
            p.drawArc(m, m, w - m*2, w - m*2, 90*16, -int(fill * 360 * 16))

        # 6. Confidence arc (fusion strength)
        if self.combined >= 0:
            p.setPen(QPen(color, 4))
            m2 = 30
            conf = abs(self.combined - 0.5) * 2
            p.drawArc(m2, m2, w - m2*2, w - m2*2, 90*16, -int(conf * 360 * 16))

        p.end()

    # ===============================================================
    #  INPUT
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
                self.model_thread.locked = True
                if self.state == self.STATE_SEARCHING:
                    self.state = self.STATE_ACQUIRING
                    self.current_color = self.COLOR_ACQUIRING
                    self.video_label.setText("VIDEO: acquiring...")
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
            self.model_thread.feature_buffer.clear()
            self.latest_visual = -1.0
            self.combined = -1.0
            self.buffer_fill = 0
            if self.state == self.STATE_LOCKED:
                self.state = self.STATE_ACQUIRING
                self.current_color = self.COLOR_ACQUIRING
                self.video_label.setText("VIDEO: acquiring...")
                self.combined_label.setText("FUSION: -")
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
        print("\nShutting down FusionScanner...")
        self.move_timer.stop()
        self.anim_timer.stop()
        self.face_timeout.stop()
        self.scanner.stop()
        self.model_thread.stop()
        self.audio_thread.stop()
        event.accept()


# ===================================================================
#  MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  FusionScanner -- Auto-Tracking Deepfake Fusion")
    print("=" * 60)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device  : {dev}")
    print(f"  Visual  : {'ENABLED' if _VISUAL_AVAILABLE else 'DISABLED'}")
    print(f"  Audio   : {'ENABLED' if _AUDIO_AVAILABLE else 'DISABLED'}")
    print("=" * 60)
    print("  Keys:")
    print("    M     = Toggle manual/auto mode")
    print("    R     = Reset frame buffer")
    print("    +/-   = Resize scope window")
    print("    Esc   = Quit")
    print("=" * 60)

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    overlay = FusionOverlay()
    overlay.show()

    sys.exit(app.exec())
