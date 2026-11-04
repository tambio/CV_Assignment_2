import os
import hashlib
from collections import Counter
from PIL import Image
DATASET = "/Users/amirorman/PycharmProjects/Asig1/dataset"
CLASSES = ["mug", "pen", "funko"]
print("Сколько файлов на класс")

counts = {}
for cls in CLASSES:
    folder = os.path.join(DATASET, cls)
    files = [f for f in os.listdir(folder) if not f.startswith(".")]
    counts[cls] = len(files)
    print(f"  {cls}: {counts[cls]}")

total = sum(counts.values())
majority = max(counts.values())
minority = min(counts.values())
ratio = majority / minority
print(f"\n  Всего: {total}")
print(f"  majority/minority = {ratio:.2f}")
if ratio > 1.5:
    print("Дисбаланс классов есть")
else:
    print("Дисбаланса нет")

print("\n")
print("Размеры картинок")

for cls in CLASSES:
    folder = os.path.join(DATASET, cls)
    files = [f for f in os.listdir(folder) if not f.startswith(".")]
    sizes = Counter()
    for f in files:
        img = Image.open(os.path.join(folder, f))
        sizes[img.size] += 1
    print(f"\n  {cls}:")
    for size, n in sizes.most_common():
        print(f"    {size}: {n} files")

print("\n")
print("Дубликаты")

for cls in CLASSES:
    folder = os.path.join(DATASET, cls)
    files = [f for f in os.listdir(folder) if not f.startswith(".")]
    hashes = {}
    duplicates = []

    for f in files:
        path = os.path.join(folder, f)
        with open(path, "rb") as fp:
            h = hashlib.md5(fp.read()).hexdigest()
        if h in hashes:
            duplicates.append((f, hashes[h]))
        else:
            hashes[h] = f

    print(f" {cls}: {len(duplicates)} дубликатов")
    for dup, orig in duplicates:
        print(f" {dup} == {orig}")

