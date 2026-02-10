#!/usr/bin/env python3
"""
bridge.py -- IPC Bridge between Electron and FusionScanner
===========================================================
Runs as a child process of Electron's main.js.
Outputs newline-delimited JSON to stdout for the renderer.
Reads JSON commands from stdin.

Usage (by Electron):
    python bridge.py

Protocol:
    STDOUT → Electron:  {"type": "detection", "visual_score": 0.72, ...}
    STDIN  ← Electron:  {"command": "set_mode", "mode": "manual"}
"""

import json
import sys
import time
import threading
import traceback
import os

# Add parent directory to path so we can import FusionScanner modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ===================================================================
#  JSON Emitter — writes to stdout for Electron to read
# ===================================================================
def emit(msg_type: str, **kwargs):
    """Send a JSON message to Electron via stdout."""
    payload = {"type": msg_type, "timestamp": time.time(), **kwargs}
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def emit_status(message: str, level: str = "info"):
    """Send a status/log message."""
    emit("status", message=message, level=level)


def emit_detection(visual_score: float, audio_score: float,
                   fusion_score: float, state: str, label: str,
                   buffer_fill: int, buffer_max: int):
    """Send a detection result."""
    emit("detection",
         visual_score=round(visual_score, 4),
         audio_score=round(audio_score, 4),
         fusion_score=round(fusion_score, 4),
         state=state,
         label=label,
         buffer_fill=buffer_fill,
         buffer_max=buffer_max)


def emit_video_frame(frame_base64: str, width: int, height: int):
    """Send a video frame as base64."""
    emit("video_frame", frame=frame_base64, width=width, height=height)


# ===================================================================
#  Command Listener — reads JSON from stdin
# ===================================================================
class CommandListener(threading.Thread):
    """Reads JSON commands from stdin in a background thread."""

    def __init__(self, scanner_ref=None):
        super().__init__(daemon=True)
        self.scanner = scanner_ref
        self.running = True

    def run(self):
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break  # stdin closed → Electron quit

                line = line.strip()
                if not line:
                    continue

                cmd = json.loads(line)
                self._handle(cmd)

            except json.JSONDecodeError:
                emit_status(f"Invalid JSON command: {line}", "warn")
            except Exception as e:
                emit_status(f"Command error: {e}", "error")

    def _handle(self, cmd: dict):
        command = cmd.get("command", "")

        if command == "quit":
            emit_status("Shutting down...")
            self.running = False
            os._exit(0)

        elif command == "set_mode":
            mode = cmd.get("mode", "auto")
            emit_status(f"Mode set to: {mode}")
            if self.scanner:
                try:
                    self.scanner.manual_mode = (mode == "manual")
                except Exception:
                    pass

        elif command == "set_threshold":
            threshold = cmd.get("threshold", 0.5)
            emit_status(f"Threshold set to: {threshold}")
            if self.scanner:
                try:
                    self.scanner.THRESHOLD = float(threshold)
                except Exception:
                    pass

        elif command == "reset_buffer":
            emit_status("Buffer reset")
            if self.scanner:
                try:
                    self.scanner.model_thread.feature_buffer.clear()
                except Exception:
                    pass

        elif command == "toggle_audio":
            enabled = cmd.get("enabled", True)
            emit_status(f"Audio pipeline: {'ON' if enabled else 'OFF'}")

        else:
            emit_status(f"Unknown command: {command}", "warn")


# ===================================================================
#  Main — Standalone mode (demo data when FusionScanner unavailable)
# ===================================================================
def run_demo_mode():
    """Emit simulated detection data for UI testing."""
    import math
    import random

    emit_status("Running in DEMO mode (FusionScanner not available)")

    t = 0
    state_cycle = ["SEARCHING", "ACQUIRING", "LOCKED", "LOCKED"]
    listener = CommandListener()
    listener.start()

    while listener.running:
        cycle_idx = (t // 20) % len(state_cycle)
        state = state_cycle[cycle_idx]

        if state == "SEARCHING":
            visual = 0.0
            audio = 0.0
            label = "UNKNOWN"
            buf_fill = 0
        elif state == "ACQUIRING":
            visual = random.uniform(0.1, 0.3)
            audio = random.uniform(0.05, 0.2)
            label = "UNKNOWN"
            buf_fill = min(t % 20, 10)
        else:
            # Alternate between REAL and FAKE
            wave = math.sin(t * 0.3) * 0.3 + 0.5
            visual = max(0, min(1, wave + random.gauss(0, 0.05)))
            audio = max(0, min(1, wave * 0.8 + random.gauss(0, 0.08)))
            label = "FAKE" if visual > 0.5 else "REAL"
            buf_fill = 10

        fusion = visual * 0.7 + audio * 0.3

        emit_detection(
            visual_score=visual,
            audio_score=audio,
            fusion_score=fusion,
            state=state,
            label=label,
            buffer_fill=buf_fill,
            buffer_max=10,
        )

        t += 1
        time.sleep(0.5)


def run_scanner_mode():
    """
    Attempt to import and run FusionScanner with IPC hooks.
    Falls back to demo mode if imports fail.
    """
    try:
        # Check if we can import the scanner dependencies
        import torch
        from FusionScanner import (
            FusionOverlay, GlobalScannerThread,
            ModelThread, AudioScannerThread,
            _VISUAL_AVAILABLE, _AUDIO_AVAILABLE,
        )

        emit_status("FusionScanner modules loaded successfully")
        emit_status(f"Visual pipeline: {'ENABLED' if _VISUAL_AVAILABLE else 'DISABLED'}")
        emit_status(f"Audio pipeline: {'ENABLED' if _AUDIO_AVAILABLE else 'DISABLED'}")
        emit_status(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

        # For full integration, FusionScanner would need to be
        # refactored to emit signals that we capture here.
        # For now, fall back to demo mode with a notice.
        emit_status("Full scanner integration requires PyQt6 runtime. Using demo mode.", "warn")
        run_demo_mode()

    except ImportError as e:
        emit_status(f"Cannot import FusionScanner: {e}", "warn")
        emit_status("Falling back to demo mode...")
        run_demo_mode()


# ===================================================================
#  Entry Point
# ===================================================================
if __name__ == "__main__":
    emit_status("Bridge starting...")

    try:
        run_scanner_mode()
    except KeyboardInterrupt:
        emit_status("Bridge interrupted")
    except Exception as e:
        emit_status(f"Bridge fatal error: {e}", "error")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
