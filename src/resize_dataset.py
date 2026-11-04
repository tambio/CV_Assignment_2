from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path("/Users/amirorman/PycharmProjects/Asig1")
SOURCE = PROJECT_ROOT / "dataset_split"
TARGET = PROJECT_ROOT / "dataset_resized"
SIZE = (224, 224)
SPLITS = ["train", "val", "test"]
CLASSES = ["mug", "pen", "funko"]


def main():
    if not SOURCE.exists():
        print(f"[error] {SOURCE} cant find")
        return

    total = 0
    for split in SPLITS:
        for cls in CLASSES:
            src_folder = SOURCE / split / cls
            dst_folder = TARGET / split / cls
            dst_folder.mkdir(parents=True, exist_ok=True)

            files = sorted([
                f for f in src_folder.iterdir()
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])

            for f in files:
                img = Image.open(f).convert("RGB")
                img_resized = img.resize(SIZE, Image.LANCZOS)
                img_resized.save(dst_folder / f.name, quality=95)
                total += 1

            print(f"{split}/{cls}: {len(files)} files resized")

    print(f"\ntotal: {total}")
    print(f"saved in: {TARGET}")


if __name__ == "__main__":
    main()