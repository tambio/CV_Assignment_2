import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path("/Users/amirorman/PycharmProjects/Asig1")
DATA_DIR = PROJECT_ROOT / "dataset_resized"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FE_WEIGHTS = OUT_DIR / "mobilenet_fe_best.pt"
FT_WEIGHTS = OUT_DIR / "mobilenet_ft_best.pt"

BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-5
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

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

print(f"Classes: {train_ds.classes}")
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


print(f"\nHere MobileNetV2 and FE weights")
model = models.mobilenet_v2(weights=None)
num_classes = len(train_ds.classes)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# FE
state = torch.load(FE_WEIGHTS, map_location="cpu")
model.load_state_dict(state)
print(f"Loaded FE weights from: {FE_WEIGHTS}")

for param in model.parameters():
    param.requires_grad = False

for block in model.features[-3:]:
    for param in block.parameters():
        param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:      {total_params:,}")
print(f"Trainable parameters:  {trainable_params:,}  lst 3 blocks + classifier")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0

print(f"\nStart Fine-tuning for {NUM_EPOCHS} epochs (LR={LR})...\n")
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / total
    train_acc = correct / total
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f" train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), FT_WEIGHTS)
        print(f" saved best FT model (val_acc={val_acc:.4f})")

elapsed = time.time() - start_time
print(f"\nTraining done in {elapsed/60:.2f} min")
print(f"Best val accuracy (FT): {best_val_acc:.4f}")
print(f"Best model saved: {FT_WEIGHTS}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Loss (Fine-tuning)")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["val_acc"], label="val")
axes[1].set_title("Accuracy (Fine-tuning)")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(OUT_DIR / "mobilenet_ft_curves.png", dpi=100)
print(f"Curves saved: {OUT_DIR / 'mobilenet_ft_curves.png'}")