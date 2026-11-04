import os
import random
import matplotlib.pyplot as plt
from PIL import Image
DATASET = "/Users/amirorman/PycharmProjects/Asig1/dataset"
CLASSES = ["mug", "pen", "funko"]

for cls in CLASSES:
    folder = os.path.join(DATASET, cls)
    files = sorted(os.listdir(folder))
    sample = random.sample(files, min(8, len(files)))

    fig, axes = plt.subplots(1, 8, figsize=(20, 3))
    for ax, fname in zip(axes, sample):
        img = Image.open(os.path.join(folder, fname))
        ax.imshow(img)
        ax.set_title(fname, fontsize=8)
        ax.axis("off")

    plt.suptitle(f"{cls} — random 8", fontsize=14)
    plt.tight_layout()

plt.show()