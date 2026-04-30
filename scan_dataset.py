from pathlib import Path
from PIL import Image

DATASET_ROOT = Path("dataset")
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}


def scan(root: Path) -> list[Path]:
    corrupt = []
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    print(f"Scanning {len(all_files)} images...")

    for path in all_files:
        try:
            with Image.open(path) as img:
                img.convert("RGB")  # same operation ImageFolder uses — catches all bad files
        except Exception as e:
            print(f"  [BAD] {path}  ({e})")
            corrupt.append(path)

    return corrupt


if __name__ == "__main__":
    corrupt = scan(DATASET_ROOT)

    if not corrupt:
        print("All images OK.")
    else:
        print(f"\nFound {len(corrupt)} corrupt file(s).")
        confirm = input("Delete them? [y/N]: ").strip().lower()
        if confirm == "y":
            for p in corrupt:
                p.unlink()
                print(f"  Deleted: {p}")
            print("Done.")
