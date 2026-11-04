import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


PROJECT_ROOT = Path("/Users/amirorman/PycharmProjects/Asig1")
OUT_DIR = PROJECT_ROOT / "outputs"
WEIGHTS = OUT_DIR / "mobilenet_ft_best.pt"

CLASS_NAMES = ["funko", "mug", "pen"]
IMG_SIZE = (224, 224)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# dev
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

def build_mobilenet(num_classes=3):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


print(f"Loading model weights: {WEIGHTS}")
model = build_mobilenet(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def predict(frame_bgr):
    # BGR 2 RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # PIL 4 trans
    pil_img = Image.fromarray(frame_rgb)
    # transf
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    # predict
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    # top1
    conf, idx = probs.max(dim=0)
    return CLASS_NAMES[idx.item()], conf.item()


print("\nOpening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] не удалось открыть камеру")
    print("Проверь: System Settings → Privacy → Camera → разреши PyCharm/Terminal")
    exit(1)

print("Camera opened. Press Q or ESC to quit.\n")

#FPS
prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] не удалось прочитать кадр")
        break

    cls, conf = predict(frame)

    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    label = f"{cls}: {conf:.2f}"
    color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.5 else (0, 0, 255)

    cv2.rectangle(frame, (10, 10), (350, 90), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    h, w = frame.shape[:2]
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    x2 = x1 + side
    y2 = y1 + side
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.imshow("CV Classifier — mug / pen / funko", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:   # Q или ESC
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")