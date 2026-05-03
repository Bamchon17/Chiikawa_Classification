from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

from .dataset import get_dataloaders
from .engine import train_one_epoch, validate
from .evaluate import evaluate
from .utils.seed import set_seed
from .utils.visualization import plot_training_curves

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR     = Path("dataset/training")
VAL_DIR       = Path("dataset/validation_set")
SAVE_DIR      = Path("models")

BATCH_SIZE    = 32
PHASE1_EPOCHS = 10   # freeze features  – train classifier only
PHASE2_EPOCHS = 20   # unfreeze tail    – fine-tune end-to-end
LR_PHASE1     = 1e-3
LR_PHASE2     = 1e-4
PATIENCE_LR   = 2    # ReduceLROnPlateau patience
PATIENCE_STOP = 5    # EarlyStopping patience
SEED          = 42
# ─────────────────────────────────────────────────────────────────────────────


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model   = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model.to(device)


def run_phase(
    phase: int,
    model: nn.Module,
    loaders: dict,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    lr: float,
    class_names: list[str],
    history: dict,
) -> None:
    if phase == 1:
        for p in model.features.parameters():
            p.requires_grad = False
        optimizer = Adam(model.classifier.parameters(), lr=lr)
    else:
        for p in model.features[-2:].parameters():
            p.requires_grad = True
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    scheduler  = ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE_LR)
    early_stop = EarlyStopping(patience=PATIENCE_STOP)
    best_loss  = float("inf")

    print(f"\n{'='*55}")
    print(f"Phase {phase} — {num_epochs} epochs  lr={lr}")
    print("="*55)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, loaders["val"], criterion, device)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        tag = ""
        if val_loss < best_loss:
            best_loss = val_loss
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "classes": class_names},
                SAVE_DIR / "best_model.pth",
            )
            tag = "  ← saved"

        print(
            f"[P{phase}] {epoch:02d}/{num_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}{tag}"
        )

        if early_stop.step(val_loss):
            print(f"Early stopping at epoch {epoch}.")
            break


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders     = get_dataloaders(TRAIN_DIR, VAL_DIR, batch_size=BATCH_SIZE)
    class_names = loaders["train"].dataset.classes  # ground truth from ImageFolder — never mismatches
    print(f"Class mapping : {loaders['train'].dataset.class_to_idx}")

    model     = build_model(num_classes=len(class_names), device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    run_phase(1, model, loaders, criterion, device, PHASE1_EPOCHS, LR_PHASE1, class_names, history)
    run_phase(2, model, loaders, criterion, device, PHASE2_EPOCHS, LR_PHASE2, class_names, history)

    checkpoint = torch.load(SAVE_DIR / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print("\n=== Final Validation Evaluation ===")
    evaluate(model, loaders["val"], device, class_names,
             save_path=str(SAVE_DIR / "confusion_matrix.png"))

    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
        save_path=str(SAVE_DIR / "training_curves.png"),
    )


if __name__ == "__main__":
    main()
