from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .utils.transforms import train_transform, val_transform

_PIN = torch.cuda.is_available()


def get_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    train_ds = ImageFolder(str(train_dir), transform=train_transform)
    val_ds   = ImageFolder(str(val_dir),   transform=val_transform)

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=_PIN),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=_PIN),
    }

    if test_dir and Path(test_dir).exists():
        test_ds = ImageFolder(str(test_dir), transform=val_transform)
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=_PIN)
        print(f"  Test : {len(test_ds)} images")

    print(f"Classes : {train_ds.classes}")
    print(f"  Train : {len(train_ds)} images")
    print(f"  Val   : {len(val_ds)} images")

    return loaders
