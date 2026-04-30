import numpy as np


def accuracy(preds: list[int], labels: list[int]) -> float:
    preds = np.array(preds)
    labels = np.array(labels)
    return float((preds == labels).mean())


def confusion_matrix(preds: list[int], labels: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        matrix[l][p] += 1
    return matrix


def classification_report(
    preds: list[int],
    labels: list[int],
    class_names: list[str],
) -> str:
    cm = confusion_matrix(preds, labels, len(class_names))
    lines = [f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"]
    lines.append("-" * 55)

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support   = cm[i, :].sum()
        lines.append(f"{name:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

    acc = accuracy(preds, labels)
    lines.append("-" * 55)
    lines.append(f"{'Accuracy':<15} {acc:>10.4f}")
    return "\n".join(lines)
