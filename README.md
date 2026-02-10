# 🎯 Live Deepfake Detection — Fusion Scanner

A real-time deepfake detection system that **automatically hunts for faces on screen**, locks onto them with a sniper-scope overlay, and runs **dual visual + audio analysis pipelines** simultaneously.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Auto-Tracking** | MTCNN-based full-screen face detection with smooth lerp tracking |
| **Visual Pipeline** | EfficientNetV2-S feature extraction → GRU temporal classifier |
| **Audio Pipeline** | Live microphone capture → AASIST anti-spoofing model |
| **Score Fusion** | Weighted combination (70% visual + 30% audio) for robust detection |
| **Sniper-Scope UI** | Transparent PyQt6 overlay with crosshairs, spinning arcs, and colour-coded feedback |
| **Manual Mode** | Press `M` to switch to manual drag-to-position mode |

### Colour States
- ⚪ **Grey** — Searching for a face
- 🟡 **Yellow** — Face acquired, filling frame buffer
- 🟢 **Green** — Fusion says **REAL**
- 🔴 **Red** — Fusion says **FAKE**

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    FusionOverlay (UI)                 │
│         Transparent sniper-scope window              │
│         Fuses visual + audio scores                  │
└──────────┬───────────────┬───────────────┬───────────┘
           │               │               │
    ┌──────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐
    │ GlobalScan  │  │ ModelThread │  │ AudioScan  │
    │ Thread      │  │            │  │ Thread     │
    │             │  │ EfficientNet│  │            │
    │ Full-screen │  │ + GRU on   │  │ PyAudio +  │
    │ MTCNN face  │  │ locked     │  │ AASIST on  │
    │ hunting     │  │ region     │  │ microphone │
    └─────────────┘  └────────────┘  └────────────┘
```

---

## 📋 Prerequisites

- **Python** 3.10 or higher
- **CUDA** (optional, recommended for GPU acceleration)
- **Microphone** for audio pipeline
- A screen with a face visible (video call, photo, etc.)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/bps-rajora/Live-Deepfake-Detection.git
cd Live-Deepfake-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install timm facenet-pytorch pyaudio
```

### 4. Download model weights

The pre-trained weights are **not included** in this repo due to file-size limits. Download them and place into `models/weights/`:

| Model | File | Destination |
|-------|------|-------------|
| EfficientNetV2-S (video) | `best_ffpp_efficientnet.pth` | `models/weights/VIDEO/` |
| GRU temporal head | `best_rnn.pt` | `models/weights/VIDEO/` |
| AASIST (audio) | `AASIST.pth` | `models/weights/AASIST/` |
| AASIST-L (audio, lighter) | `AASIST-L.pth` | `models/weights/AASIST/` |

> **Note:** The application will still launch if weights are missing — the corresponding pipeline will simply show "OFF".

### 5. Download MediaPipe model files

Place these in the project root:
- `face_landmarker.task` — [Download from Google](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- `blaze_face_short_range.tflite` — [Download from Google](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite)

### 6. Run

```bash
# Full Fusion Scanner (visual + audio)
python FusionScanner.py

# Visual-only Auto Tracker
python AutoTracker.py
```

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `M` | Toggle Manual / Auto tracking mode |
| `+` / `-` | Resize the scanner overlay |
| `R` | Reset the frame buffer |
| `Esc` | Quit |
| **Drag** | Move the overlay (Manual mode) |

---

## 📁 Project Structure

```
.
├── FusionScanner.py        # Main app: visual + audio fusion scanner
├── AutoTracker.py          # Visual-only auto-tracking scanner
├── AASISTMODEL.py          # AASIST model wrapper for live audio inference
├── realtime_inference.py   # Standalone real-time inference script
├── main.py                 # AASIST training / evaluation entry point
├── models/
│   ├── AASIST.py           # AASIST model architecture
│   ├── RawNet2Spoof.py     # RawNet2 baseline model
│   ├── RawNetGatSpoofST.py # RawGAT-ST baseline model
│   └── weights/            # Pre-trained weights (not tracked by git)
│       ├── AASIST/
│       └── VIDEO/
├── config/                 # Training config files (.conf)
├── requirements.txt
├── LICENSE                 # MIT License
└── NOTICE                  # Third-party attributions
```

---

## 🙏 Acknowledgements

This project builds upon:

- **[AASIST](https://github.com/clovaai/aasist)** — Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks (NAVER Corp.)
- **[ASVspoof 2019](https://www.asvspoof.org/)** — Large-scale public database of synthesized, converted, and replayed speech
- **[EfficientNetV2](https://github.com/huggingface/pytorch-image-models)** — via `timm` library
- **[MTCNN](https://github.com/timesler/facenet-pytorch)** — via `facenet-pytorch`

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

Original AASIST code: Copyright (c) 2021-present NAVER Corp. — see [NOTICE](NOTICE).
