import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils.metrics import confusion_matrix, classification_report
from .utils.visualization import plot_confusion_matrix


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    plot: bool = True,
    save_path: str | None = None,
) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_preds, all_labels, len(class_names))
    print("\n" + classification_report(all_preds, all_labels, class_names))

    if plot:
        plot_confusion_matrix(cm, class_names, save_path=save_path)

    return {"preds": all_preds, "labels": all_labels, "confusion_matrix": cm}
