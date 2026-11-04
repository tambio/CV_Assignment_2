import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import time
import os

DATA_DIR = "/Users/amirorman/PycharmProjects/Asig1/dataset_resized"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
BATCH_SIZE = 32

# trans
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_ds = datasets.ImageFolder(f"{DATA_DIR}/test", transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


def evaluate_model(model, name):
    model.eval()
    all_preds, all_labels = [], []

    start_time = time.time()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    end_time = time.time()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    latency = (end_time - start_time) / len(test_ds) * 1000  # ms per image

    print(f"\n {name} ")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"F1-Score (macro): {f1:.4f}")
    print(f"Avg Latency: {latency:.2f} ms/image")


#baseline mobnetv2
#srch file
path_mobile = "../outputs/mobilenet_ft_best.pt"
if os.path.exists(path_mobile):
    print("Here ll be MobileNetV2 (A1 Baseline)")
    mobilenet = models.mobilenet_v2()
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 3)
    mobilenet.load_state_dict(torch.load(path_mobile, map_location=DEVICE))
    mobilenet = mobilenet.to(DEVICE)
    evaluate_model(mobilenet, "A1 Baseline: MobileNetV2")
else:
    print(f"File not found: {path_mobile}")

#sota effnetB0
path_sota = "efficientnet_sota.pth"
if os.path.exists(path_sota):
    print("\n Here ll be EfficientNet-B0 (A2 SOTA)")
    efficientnet = models.efficientnet_b0()
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 3)
    efficientnet.load_state_dict(torch.load(path_sota, map_location=DEVICE))
    efficientnet = efficientnet.to(DEVICE)
    evaluate_model(efficientnet, "A2 SOTA: EfficientNet-B0")
else:
    print(f"File not found: {path_sota}")

#ablation effnet no arg
path_ablation = "efficientnet_no_aug.pth"
if os.path.exists(path_ablation):
    print("\nHere ll be EfficientNet-B0 (Ablation No Aug)")
    eff_noaug = models.efficientnet_b0()
    eff_noaug.classifier[1] = nn.Linear(eff_noaug.classifier[1].in_features, 3)
    eff_noaug.load_state_dict(torch.load(path_ablation, map_location=DEVICE))
    eff_noaug = eff_noaug.to(DEVICE)
    evaluate_model(eff_noaug, "A2 Ablation: EfficientNet (No Aug)")
else:
    print(f"File not found: {path_ablation}")