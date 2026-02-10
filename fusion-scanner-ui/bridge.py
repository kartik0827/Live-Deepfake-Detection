#!/usr/bin/env python3
"""
bridge.py -- Real IPC Bridge between Electron and FusionScanner
================================================================
Runs as a headless child process of Electron's main.js.
Spawns the actual detection threads (GlobalScanner, ModelThread,
AudioScanner) and forwards all results as JSON via stdout.

Protocol:
    STDOUT → Electron:  {"type": "detection", "visual_score": 0.72, ...}
    STDIN  ← Electron:  {"command": "set_mode", "mode": "manual"}
"""

import json
import sys
import os
import time
import threading
import traceback
import base64

# ── Add project root to path ─────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_ROOT)


# ===================================================================
#  JSON Emitters — write to stdout for Electron to read
# ===================================================================
def emit(msg_type: str, **kwargs):
    """Send a JSON message to Electron via stdout."""
    payload = {"type": msg_type, "timestamp": time.time(), **kwargs}
    try:
        line = json.dumps(payload)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def emit_status(message: str, level: str = "info"):
    emit("status", message=message, level=level)


def emit_detection(visual_score: float, audio_score: float,
                   fusion_score: float, state: str, label: str,
                   buffer_fill: int, buffer_max: int):
    emit("detection",
         visual_score=round(visual_score, 4),
         audio_score=round(audio_score, 4),
         fusion_score=round(fusion_score, 4),
         state=state,
         label=label,
         buffer_fill=buffer_fill,
         buffer_max=buffer_max)


def emit_video_frame(frame_base64: str, width: int, height: int):
    emit("video_frame", frame=frame_base64, width=width, height=height)


# ===================================================================
#  Command Listener — reads JSON from stdin
# ===================================================================
class CommandListener(threading.Thread):
    """Reads JSON commands from stdin in a background thread."""

    def __init__(self, scanner=None):
        super().__init__(daemon=True)
        self.scanner = scanner
        self.running = True

    def run(self):
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                cmd = json.loads(line)
                self._handle(cmd)
            except json.JSONDecodeError:
                emit_status(f"Invalid JSON from stdin", "warn")
            except Exception as e:
                emit_status(f"Command error: {e}", "error")

    def _handle(self, cmd: dict):
        command = cmd.get("command", "")
        scanner = self.scanner

        if command == "quit":
            emit_status("Shutting down bridge...")
            self.running = False
            if scanner:
                scanner.shutdown()
            os._exit(0)

        elif command == "set_mode":
            mode = cmd.get("mode", "auto")
            emit_status(f"Mode: {mode}")
            if scanner:
                scanner.set_mode(mode)

        elif command == "set_threshold":
            threshold = float(cmd.get("threshold", 0.5))
            emit_status(f"Threshold: {threshold}")
            if scanner:
                scanner.threshold = threshold

        elif command == "reset_buffer":
            emit_status("Buffer reset")
            if scanner:
                scanner.reset_buffer()

        elif command == "toggle_audio":
            enabled = cmd.get("enabled", True)
            emit_status(f"Audio pipeline: {'ON' if enabled else 'OFF'}")
            if scanner:
                scanner.toggle_audio(enabled)

        else:
            emit_status(f"Unknown command: {command}", "warn")


# ===================================================================
#  HeadlessScanner — Runs real detection threads without GUI
# ===================================================================
class HeadlessScanner:
    """
    Headless version of FusionOverlay that runs the actual detection
    threads and emits results via JSON stdout instead of painting UI.
    """

    STATE_SEARCHING = "SEARCHING"
    STATE_ACQUIRING = "ACQUIRING"
    STATE_LOCKED    = "LOCKED"

    VISUAL_WEIGHT = 0.7
    AUDIO_WEIGHT  = 0.3

    def __init__(self):
        self.state = self.STATE_SEARCHING
        self.threshold = 0.5
        self.manual_mode = False
        self.audio_enabled = True

        # Scores
        self.latest_visual = -1.0
        self.latest_audio  = -1.0
        self.combined      = -1.0
        self.buffer_fill   = 0
        self.buffer_max    = 10

        # Tracking geometry (for headless, we track face position)
        self.window_size = 340
        self.target_x = 0
        self.target_y = 0
        self.current_x = 0
        self.current_y = 0

        # Thread references
        self.scanner_thread = None
        self.model_thread   = None
        self.audio_thread   = None
        self.face_lost_timer = None
        self._face_timer_lock = threading.Lock()

        # Video frame capture
        self._capture_lock = threading.Lock()
        self._frame_counter = 0

    def start(self):
        """Initialize and start all detection threads."""
        from PyQt6.QtCore import QTimer

        # Import scanner threads from FusionScanner
        from FusionScanner import (
            GlobalScannerThread, ModelThread, AudioScannerThread,
            _VISUAL_AVAILABLE, _AUDIO_AVAILABLE,
        )

        emit_status(f"Visual pipeline: {'ENABLED' if _VISUAL_AVAILABLE else 'DISABLED'}")
        emit_status(f"Audio pipeline: {'ENABLED' if _AUDIO_AVAILABLE else 'DISABLED'}")

        # ── 1. Global face scanner ──────────────────────────────
        self.scanner_thread = GlobalScannerThread()
        self.scanner_thread.face_found.connect(self._on_face_found)
        self.scanner_thread.no_face.connect(self._on_no_face)
        self.scanner_thread.start()
        emit_status("GlobalScanner started")

        # ── 2. Visual model thread ──────────────────────────────
        self.model_thread = ModelThread(
            capture_rect_fn=self._get_capture_rect,
        )
        self.model_thread.prediction.connect(self._on_visual_score)
        self.model_thread.buffer_status.connect(self._on_buffer_status)
        self.model_thread.status_msg.connect(
            lambda msg: emit_status(f"Visual: {msg}")
        )
        self.model_thread.start()
        emit_status("ModelThread started")

        # ── 3. Audio model thread ───────────────────────────────
        if _AUDIO_AVAILABLE:
            self.audio_thread = AudioScannerThread()
            self.audio_thread.audio_score.connect(self._on_audio_score)
            self.audio_thread.status_msg.connect(
                lambda msg: emit_status(f"Audio: {msg}")
            )
            self.audio_thread.start()
            emit_status("AudioScanner started")
        else:
            emit_status("Audio pipeline disabled (dependencies missing)", "warn")

        # ── 4. Face lost timeout timer ──────────────────────────
        self.face_lost_timer = QTimer()
        self.face_lost_timer.setSingleShot(True)
        self.face_lost_timer.timeout.connect(self._on_face_lost_timeout)

        # ── 5. Periodic video frame emitter ─────────────────────
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._emit_video_frame)
        self.frame_timer.start(200)  # ~5 FPS

        # ── 6. Periodic position update ─────────────────────────
        self.pos_timer = QTimer()
        self.pos_timer.timeout.connect(self._update_position)
        self.pos_timer.start(16)

        emit_status("All pipelines started — scanner is LIVE")
        self._emit_state()

    # ── Face Tracking Slots ─────────────────────────────────────
    def _on_face_found(self, x, y, w, h):
        if self.manual_mode:
            return

        self.target_x = x + (w - self.window_size) // 2
        self.target_y = y + (h - self.window_size) // 2
        self.face_lost_timer.start(1500)

        if self.state == self.STATE_SEARCHING:
            self.state = self.STATE_ACQUIRING
            self.model_thread.locked = True
            self.latest_visual = -1.0
            self._emit_state()

    def _on_no_face(self):
        pass  # Wait for timeout

    def _on_face_lost_timeout(self):
        if self.manual_mode:
            return
        self.state = self.STATE_SEARCHING
        if self.model_thread:
            self.model_thread.locked = False
        self.latest_visual = -1.0
        self.combined = -1.0
        self.buffer_fill = 0
        self._emit_state()

    # ── Visual Score Slot ───────────────────────────────────────
    def _on_visual_score(self, prob):
        self.latest_visual = prob
        self.state = self.STATE_LOCKED
        self._recalculate()

    def _on_buffer_status(self, current, total):
        self.buffer_fill = current
        self.buffer_max = total

    # ── Audio Score Slot ────────────────────────────────────────
    def _on_audio_score(self, prob):
        self.latest_audio = prob
        self._recalculate()

    # ── Fusion Calculation ──────────────────────────────────────
    def _recalculate(self):
        v = self.latest_visual
        a = self.latest_audio

        if v >= 0 and a >= 0:
            self.combined = v * self.VISUAL_WEIGHT + a * self.AUDIO_WEIGHT
        elif v >= 0:
            self.combined = v
        elif a >= 0:
            self.combined = a
        else:
            return

        label = "FAKE" if self.combined > self.threshold else "REAL"
        self._emit_state(label=label)

    def _emit_state(self, label=None):
        """Emit the current detection state as JSON."""
        if label is None:
            if self.combined >= 0:
                label = "FAKE" if self.combined > self.threshold else "REAL"
            else:
                label = "UNKNOWN"

        emit_detection(
            visual_score=max(0, self.latest_visual),
            audio_score=max(0, self.latest_audio),
            fusion_score=max(0, self.combined),
            state=self.state,
            label=label,
            buffer_fill=self.buffer_fill,
            buffer_max=self.buffer_max,
        )

    # ── Position Update (LERP) ──────────────────────────────────
    def _update_position(self):
        if self.manual_mode:
            return
        s = 0.25
        self.current_x = int(self.current_x * (1 - s) + self.target_x * s)
        self.current_y = int(self.current_y * (1 - s) + self.target_y * s)

    def _get_capture_rect(self):
        return {
            "left":   self.current_x,
            "top":    self.current_y,
            "width":  self.window_size,
            "height": self.window_size,
        }

    # ── Video Frame Emitter ─────────────────────────────────────
    def _emit_video_frame(self):
        """Capture the current tracked region and send as base64."""
        self._frame_counter += 1
        if self._frame_counter % 2 != 0:  # Send every other frame to reduce bandwidth
            return

        try:
            import cv2
            import numpy as np
            from mss import mss

            rect = self._get_capture_rect()
            with mss() as sct:
                screenshot = sct.grab(rect)
                frame = np.array(screenshot)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Resize for bandwidth efficiency
                frame_small = cv2.resize(frame_bgr, (320, 320))

                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame_small, [
                    cv2.IMWRITE_JPEG_QUALITY, 60
                ])
                b64 = base64.b64encode(buffer).decode('utf-8')

                emit_video_frame(b64, 320, 320)
        except Exception:
            pass  # Silently skip frame errors

    # ── Commands from Electron ──────────────────────────────────
    def set_mode(self, mode: str):
        self.manual_mode = (mode == "manual")
        if self.manual_mode:
            if self.model_thread:
                self.model_thread.locked = True
            if self.state == self.STATE_SEARCHING:
                self.state = self.STATE_ACQUIRING
                self._emit_state()
        emit_status(f"Tracking mode: {'MANUAL' if self.manual_mode else 'AUTO'}")

    def reset_buffer(self):
        if self.model_thread:
            self.model_thread.feature_buffer.clear()
        self.latest_visual = -1.0
        self.combined = -1.0
        self.buffer_fill = 0
        if self.state == self.STATE_LOCKED:
            self.state = self.STATE_ACQUIRING
        self._emit_state()

    def toggle_audio(self, enabled: bool):
        self.audio_enabled = enabled
        if not enabled and self.audio_thread:
            self.audio_thread.stop()
            self.latest_audio = -1.0
            emit_status("Audio pipeline stopped")
        elif enabled and self.audio_thread is None:
            try:
                from FusionScanner import AudioScannerThread
                self.audio_thread = AudioScannerThread()
                self.audio_thread.audio_score.connect(self._on_audio_score)
                self.audio_thread.status_msg.connect(
                    lambda msg: emit_status(f"Audio: {msg}")
                )
                self.audio_thread.start()
                emit_status("Audio pipeline restarted")
            except Exception as e:
                emit_status(f"Failed to restart audio: {e}", "error")

    def shutdown(self):
        """Gracefully stop all threads."""
        emit_status("Shutting down all threads...")
        if self.face_lost_timer:
            self.face_lost_timer.stop()
        if hasattr(self, 'frame_timer'):
            self.frame_timer.stop()
        if hasattr(self, 'pos_timer'):
            self.pos_timer.stop()
        if self.scanner_thread:
            self.scanner_thread.stop()
        if self.model_thread:
            self.model_thread.stop()
        if self.audio_thread:
            self.audio_thread.stop()
        emit_status("All threads stopped")


# ===================================================================
#  Main Entry Point
# ===================================================================
def run_real_scanner():
    """Run the real detection pipeline with a headless Qt event loop."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QCoreApplication
    import torch

    emit_status("Initializing real detection pipeline...")
    emit_status(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Create headless Qt application (needed for QThread/QTimer/signals)
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create and start the headless scanner
    scanner = HeadlessScanner()

    # Start command listener (stdin reader)
    listener = CommandListener(scanner=scanner)
    listener.start()

    # Start all detection threads
    scanner.start()

    # Run the Qt event loop (keeps threads and timers alive)
    sys.exit(app.exec())


def run_demo_mode():
    """Fallback: emit simulated data when dependencies are missing."""
    import math
    import random

    emit_status("Running in DEMO mode (real models not available)")

    listener = CommandListener()
    listener.start()

    t = 0
    while listener.running:
        cycle = t % 40
        if cycle < 8:
            state, visual, audio, label, buf = "SEARCHING", 0.0, 0.0, "UNKNOWN", 0
        elif cycle < 14:
            state = "ACQUIRING"
            visual = random.uniform(0.1, 0.3)
            audio = random.uniform(0.05, 0.2)
            label, buf = "UNKNOWN", min(t % 20, 10)
        else:
            state = "LOCKED"
            wave = math.sin(t * 0.3) * 0.3 + 0.5
            visual = max(0, min(1, wave + random.gauss(0, 0.05)))
            audio = max(0, min(1, wave * 0.8 + random.gauss(0, 0.08)))
            label = "FAKE" if visual > 0.5 else "REAL"
            buf = 10

        fusion = visual * 0.7 + audio * 0.3
        emit_detection(visual, audio, fusion, state, label, buf, 10)
        t += 1
        time.sleep(0.5)


if __name__ == "__main__":
    emit_status("Bridge starting...")

    try:
        # Try to import all required dependencies
        import torch
        from PyQt6.QtWidgets import QApplication
        from FusionScanner import (
            GlobalScannerThread, ModelThread, AudioScannerThread,
            _VISUAL_AVAILABLE, _AUDIO_AVAILABLE,
        )
        emit_status("All dependencies loaded — starting real scanner")
        run_real_scanner()

    except ImportError as e:
        emit_status(f"Missing dependency: {e}", "warn")
        emit_status("Falling back to demo mode...")
        run_demo_mode()

    except Exception as e:
        emit_status(f"Fatal error: {e}", "error")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
