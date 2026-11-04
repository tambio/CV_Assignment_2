import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os

# 1. Настройки
DATA_DIR = "/Users/amirorman/PycharmProjects/Asig1/dataset_resized"
BATCH_SIZE = 32
NUM_EPOCHS = 5
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# 2. Трансформации
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загружаем данные
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes: {train_ds.classes}")
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# 3. Модель EfficientNet-B0
print("\nLoading EfficientNet-B0...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Списки для хранения метрик (чтобы потом нарисовать графики)
history_train_loss = []
history_val_acc = []

# 4. Обучение
print("Start training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    history_train_loss.append(epoch_train_loss)

    # Валидация
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_val_acc = correct / total
    history_val_acc.append(epoch_val_acc)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

# 5. Сохраняем модель
torch.save(model.state_dict(), "efficientnet_sota.pth")
print("\nModel saved as efficientnet_sota.pth")

# 6. Рисуем графики на основе РЕАЛЬНЫХ данных
os.makedirs("../outputs", exist_ok=True)
epochs = list(range(1, NUM_EPOCHS + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График Loss
axes[0].plot(epochs, history_train_loss, marker='o', color='blue')
axes[0].set_title("EfficientNet-B0: Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

# График Accuracy
axes[1].plot(epochs, history_val_acc, marker='s', color='green')
axes[1].set_title("EfficientNet-B0: Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("../outputs/sota_training_curves.png", dpi=150)
print("Training curves saved to outputs/sota_training_curves.png")
plt.show()