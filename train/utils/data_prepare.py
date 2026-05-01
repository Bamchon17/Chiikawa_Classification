import random
import shutil
from pathlib import Path

SEED = 42
VAL_RATIO = 0.10

TRAIN_DIR = Path("dataset/training")
VAL_DIR = Path("dataset/validation_set")


def split_class(class_name: str, dry_run: bool = False) -> dict:
    src = TRAIN_DIR / class_name
    dst = VAL_DIR / class_name

    files = sorted([
        f for f in src.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    if not files:
        print(f"[WARN] {class_name}: no images found in {src}")
        return {}

    random.seed(SEED)
    random.shuffle(files)

    n_val = max(1, round(len(files) * VAL_RATIO))
    val_files = files[:n_val]
    train_remaining = files[n_val:]

    print(f"\n{class_name}:")
    print(f"  Total : {len(files)}")
    print(f"  Train : {len(train_remaining)}")
    print(f"  Val   : {len(val_files)}")

    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        for f in val_files:
            shutil.move(str(f), dst / f.name)
        print(f"  Moved {len(val_files)} files to {dst}")

    return {"class": class_name, "total": len(files), "train": len(train_remaining), "val": len(val_files)}


def verify():
    print("\n=== Verification ===")
    for cls_dir in sorted(TRAIN_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        train_count = sum(1 for f in cls_dir.iterdir() if f.is_file())
        val_dir = VAL_DIR / cls_dir.name
        val_count = sum(1 for f in val_dir.iterdir() if f.is_file()) if val_dir.exists() else 0
        total = train_count + val_count
        ratio = val_count / total * 100 if total else 0
        print(f"  {cls_dir.name}: train={train_count}, val={val_count}, val%={ratio:.1f}%")

        # Check for overlap
        train_names = {f.name for f in cls_dir.iterdir() if f.is_file()}
        val_names = {f.name for f in val_dir.iterdir() if f.is_file()} if val_dir.exists() else set()
        overlap = train_names & val_names
        if overlap:
            print(f"  [ERROR] Data leakage detected! {len(overlap)} overlapping files: {list(overlap)[:5]}")
        else:
            print(f"  [OK] No data leakage")


if __name__ == "__main__":
    classes = [d.name for d in TRAIN_DIR.iterdir() if d.is_dir()]
    print(f"Classes found: {classes}")

    # Dry run first
    print("\n--- Dry Run (no files moved) ---")
    for cls in sorted(classes):
        split_class(cls, dry_run=True)

    confirm = input("\nProceed with moving files? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        exit(0)

    print("\n--- Moving files ---")
    for cls in sorted(classes):
        split_class(cls, dry_run=False)

    verify()
    print("\nDone.")
