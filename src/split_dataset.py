import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/Users/amirorman/PycharmProjects/Asig1")
SOURCE = PROJECT_ROOT / "dataset"
TARGET = PROJECT_ROOT / "dataset_split"
CLASSES = ["mug", "pen", "funko"]

SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

def main():
    random.seed(RANDOM_SEED)

    if TARGET.exists():
        print(f"[warning] {TARGET} уже существует.")
        ans = input("Удалить и пересоздать? (у или п): ").strip().lower()
        if ans == "y":
            shutil.rmtree(TARGET)
        else:
            print("Отменено.")
            return

    for split_name in SPLIT:
        for cls in CLASSES:
            (TARGET / split_name / cls).mkdir(parents=True, exist_ok=True)

    print(f"Источник: {SOURCE}")
    print(f"Назначение: {TARGET}\n")

    for cls in CLASSES:
        src_folder = SOURCE / cls
        files = sorted([
            f for f in src_folder.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        random.shuffle(files)
        n = len(files)

        n_train = int(n * SPLIT["train"])
        n_val = int(n * SPLIT["val"])
        n_test = n - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for f in train_files:
            shutil.copy2(f, TARGET / "train" / cls / f.name)
        for f in val_files:
            shutil.copy2(f, TARGET / "val" / cls / f.name)
        for f in test_files:
            shutil.copy2(f, TARGET / "test" / cls / f.name)

        print(f"{cls}: total={n}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    print("\nГотово.")
    print(TARGET)


if __name__ == "__main__":
    main()