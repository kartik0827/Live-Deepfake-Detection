'''import cv2
import torch
from collections import deque
import timm
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms

# -------------------------------
# STAGE 1: DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# STAGE 2: LOAD TRAINED MODEL
# -------------------------------
model = timm.create_model(
    "tf_efficientnetv2_s",
    pretrained=False,
    num_classes=1
)

model.load_state_dict(
    torch.load("best_ffpp_efficientnet.pth", map_location=device)
)
model = model.to(device)
model.eval()

print("EfficientNet model loaded")

# -------------------------------
# STAGE 3: PREPROCESSING (MATCH TRAINING)
# -------------------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# STAGE 4: MTCNN (FACE DETECTOR)
# -------------------------------
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    device=device
)

print("MTCNN initialized")

# -------------------------------
# STAGE 5: FRAME PREDICTION FUNCTION
# -------------------------------
@torch.no_grad()
def predict_frame(frame_bgr):
    """
    Input: BGR frame from OpenCV
    Output: probability (0–1) or None
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    face = mtcnn(rgb)
    if face is None:
        return None

    face = face.unsqueeze(0).to(device)
    logits = model(face)
    prob = torch.sigmoid(logits).item()

    return prob

# -------------------------------
# STAGE 6: VIDEO / WEBCAM LOOP
# -------------------------------

# OPTION 1: Webcam (default)
cap = cv2.VideoCapture("ankit.mp4")


# OPTION 2: Video file
# cap = cv2.VideoCapture("test_video.mp4")

frame_id = 0
THRESHOLD = 0.5

print("Starting real-time inference... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # process every 10th frame (speed + stability)
    if frame_id % 15 != 0:

        continue

    prob = predict_frame(frame)

    if prob is not None:
        label = "FAKE" if prob > THRESHOLD else "REAL"
        color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)

        text = f"{label} ({prob:.2f})"
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    cv2.imshow("Deepfake Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()'''
#the above code is more enhanced
import cv2
import torch
import timm
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms
from collections import deque

# -------------------------------
# DEVICE (CPU SAFE)
# -------------------------------
device = torch.device("cpu")
print("Using device:", device)

# -------------------------------
# LOAD EFFICIENTNET (FEATURE EXTRACTOR)
# -------------------------------
cnn = timm.create_model(
    "tf_efficientnetv2_s",
    pretrained=False,
    num_classes=0  # IMPORTANT: feature extractor
)

cnn.load_state_dict(
    torch.load("best_ffpp_efficientnet.pth", map_location=device),
    strict=False
)
cnn = cnn.to(device)
cnn.eval()

print("EfficientNet loaded as feature extractor")

# -------------------------------
# LOAD LSTM / GRU MODEL
# -------------------------------
import torch.nn as nn

import torch.nn as nn

class DeepfakeGRU(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(          # 🔑 name MUST be rnn
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        out = self.fc(h_n[-1])
        return out

rnn = DeepfakeGRU(
    input_dim=1280,
    hidden_dim=256,
    num_layers=1
)

rnn.load_state_dict(
    torch.load("best_rnn.pt", map_location=device)
)

rnn = rnn.to(device)
rnn.eval()

print("Temporal LSTM model loaded")

# -------------------------------
# PREPROCESSING
# -------------------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# MTCNN
# -------------------------------
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    device=device
)

# -------------------------------
# TEMPORAL BUFFER
# -------------------------------
SEQ_LEN = 15
feature_buffer = deque(maxlen=SEQ_LEN)
time_buffer = deque(maxlen=SEQ_LEN)

THRESHOLD = 0.6

# -------------------------------
# FRAME → FEATURE
# -------------------------------
@torch.no_grad()
def extract_feature(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    if face is None:
        return None

    face = face.unsqueeze(0).to(device)
    feat = cnn(face)           # (1, 1280)
    return feat.squeeze(0)     # (1280)

# -------------------------------
# VIDEO INFERENCE
# -------------------------------
cap = cv2.VideoCapture("ankit.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

frame_id = 0
print("Starting LSTM temporal inference")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # skip frames for speed
    if frame_id % 10 != 0:
        continue

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    feat = extract_feature(frame)
    if feat is None:
        continue

    feature_buffer.append(feat)
    time_buffer.append(timestamp)

    # wait until sequence is full
    if len(feature_buffer) < SEQ_LEN:
        continue

    seq = torch.stack(list(feature_buffer)).unsqueeze(0).to(device)
    # shape: (1, T, 1280)

    with torch.no_grad():
        logit = rnn(seq)
        prob = torch.sigmoid(logit).item()

    label = "FAKE" if prob > THRESHOLD else "REAL"
    color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)

    text = f"{label} | p={prob:.2f} | t={timestamp:.2f}s"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    if label == "FAKE":
        print(f"[FAKE] {timestamp:.2f}s | prob={prob:.2f}")

    cv2.imshow("Deepfake Detection (LSTM)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
