from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

PROJECT_ROOT = Path("/Users/amirorman/PycharmProjects/Asig1")
DATA_DIR = PROJECT_ROOT / "dataset_resized"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
SEED = 42

# dev
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

torch.manual_seed(SEED)
if DEVICE.type == "mps":
    torch.mps.manual_seed(SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Classes: {test_ds.classes}")
print(f"Test size: {len(test_ds)}")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_mobilenet(num_classes=3):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def predict_all(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def evaluate_model(name, model, loader, class_names):
    print(f"\n")
    print(f"Evaluating: {name}")

    preds, labels = predict_all(model, loader)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}  (macro)")
    print(f"Recall   : {rec:.4f}  (macro)")
    print(f"F1-score : {f1:.4f}  (macro)")

    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(labels, preds)

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "preds": preds,
        "labels": labels,
    }


print("\nLoading Simple CNN")
simple_cnn = SimpleCNN(num_classes=3).to(DEVICE)
simple_cnn.load_state_dict(torch.load(OUT_DIR / "simple_cnn_best.pt", map_location=DEVICE))
simple_cnn.eval()

print("Loading MobileNetV2 (FE)")
mobilenet_fe = build_mobilenet(num_classes=3).to(DEVICE)
mobilenet_fe.load_state_dict(torch.load(OUT_DIR / "mobilenet_fe_best.pt", map_location=DEVICE))
mobilenet_fe.eval()

print("Loading MobileNetV2 (FT)")
mobilenet_ft = build_mobilenet(num_classes=3).to(DEVICE)
mobilenet_ft.load_state_dict(torch.load(OUT_DIR / "mobilenet_ft_best.pt", map_location=DEVICE))
mobilenet_ft.eval()


results = [
    evaluate_model("Simple CNN", simple_cnn, test_loader, test_ds.classes),
    evaluate_model("MobileNetV2 (FE)", mobilenet_fe, test_loader, test_ds.classes),
    evaluate_model("MobileNetV2 (FT)", mobilenet_ft, test_loader, test_ds.classes),
]


print("\n")
print("SUMMARY")
print(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
print("\n")
for r in results:
    print(f"{r['name']:<25} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
          f"{r['recall']:>8.4f} {r['f1']:>8.4f}")

# sv in txtshka
with open(OUT_DIR / "summary.txt", "w") as f:
    f.write(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}\n")
    f.write("-" * 70 + "\n")
    for r in results:
        f.write(f"{r['name']:<25} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
                f"{r['recall']:>8.4f} {r['f1']:>8.4f}\n")
print(f"\nSummary saved: {OUT_DIR / 'summary.txt'}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, r in zip(axes, results):
    sns.heatmap(
        r["confusion_matrix"],
        annot=True, fmt="d", cmap="Blues",
        xticklabels=test_ds.classes,
        yticklabels=test_ds.classes,
        ax=ax, cbar=False
    )
    ax.set_title(f"{r['name']}\nAcc={r['accuracy']:.3f}, F1={r['f1']:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrices.png", dpi=100)
print(f"Confusion matrices saved: {OUT_DIR / 'confusion_matrices.png'}")