# evaluate.py
import argparse
import os
import csv
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from model import build_model
from datasets import get_datasets, get_data_loaders

# ----------------------------
# Utils: safe checkpoint loader
# ----------------------------
def load_checkpoint_safe(path, device):
    """Load checkpoint while handling PyTorch 2.6+ weights_only safety.
    Returns the loaded object (dict or state_dict).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ck = torch.load(path, map_location=device)
        return ck
    except Exception as e:
        # Attempt allowlist for CrossEntropyLoss if error indicates weights_only issue
        err = str(e)
        if "Weights only load failed" in err or "WeightsUnpicklingError" in err or "weights_only" in err:
            torch.serialization.add_safe_globals([torch.nn.modules.loss.CrossEntropyLoss])
            ck = torch.load(path, map_location=device, weights_only=False)
            return ck
        else:
            raise

# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_paths = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Support dataloaders that return (images, labels) or (images, labels, paths)
            if len(batch) == 2:
                images, labels = batch
                paths = ["" for _ in range(images.size(0))]
            elif len(batch) >= 3:
                images, labels, paths = batch[:3]
            else:
                raise RuntimeError("Unexpected batch format from dataloader. Expect (images, labels) or (images, labels, paths).")

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # logits
            probs = softmax(outputs).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds.tolist())
            # store probability of predicted class (or full prob vector if you prefer)
            all_probs.extend(probs.tolist())
            all_labels.extend(labels_np.tolist())
            # convert paths -> strings (if provided)
            all_paths.extend([p if isinstance(p, str) else str(p) for p in paths])

    return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_paths

# ----------------------------
# Plot confusion matrix
# ----------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, out_path, figsize=(8, 6)):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.0)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                     xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return cm

# ----------------------------
# Main CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model: confusion matrix + classification report")
    parser.add_argument('--checkpoint', type=str, default="../outputs/model_pretrained_True.pth",
                        help='Path to checkpoint (.pth)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Whether model was created with pretrained backbone (affects dataset call)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--out-dir', type=str, default="../outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load datasets and loaders (reuse your functions)
    print("[INFO] Loading datasets...")
    dataset_train, dataset_valid, dataset_classes = get_datasets(args.pretrained)
    # get_data_loaders probably returns (train_loader, valid_loader) — try to use batch_size
    try:
        train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size=args.batch_size)
    except TypeError:
        # fallback: assume your get_data_loaders ignores batch_size; just call without
        train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Classes: {dataset_classes}")

    # Build model architecture (must match training)
    model = build_model(pretrained=False, fine_tune=False, num_classes=len(dataset_classes))
    model = model.to(device)

    # Load checkpoint safely and extract state_dict
    checkpoint = load_checkpoint_safe(args.checkpoint, device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip 'module.' prefix if needed
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith('module.') and not any(k.startswith('module.') for k in model.state_dict().keys()):
        new_state = {}
        for k, v in state_dict.items():
            new_state[k.replace('module.', '', 1)] = v
        state_dict = new_state

    print("[INFO] Loading model weights...")
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluate on validation set
    y_true, y_pred, y_probs, paths = evaluate(model, valid_loader, device, dataset_classes)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"[RESULT] Overall accuracy: {accuracy * 100:.2f}%")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=dataset_classes, digits=4)
    print("Classification report:")
    print(report)

    # Save classification report to text
    report_path = os.path.join(args.out_dir, f"classification_report_pretrained_{args.pretrained}.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall accuracy: {accuracy * 100:.4f}%\n\n")
        f.write(report)
    print(f"[SAVED] Classification report -> {report_path}")

    # Save predictions CSV
    csv_path = os.path.join(args.out_dir, f"predictions_pretrained_{args.pretrained}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # header
        header = ["filepath", "true_label", "pred_label"] + [f"prob_class_{c}" for c in dataset_classes]
        writer.writerow(header)
        for p, t, pr, probs in zip(paths, y_true.tolist(), y_pred.tolist(), y_probs.tolist()):
            true_name = dataset_classes[int(t)]
            pred_name = dataset_classes[int(pr)]
            writer.writerow([p, true_name, pred_name] + probs)
    print(f"[SAVED] Predictions CSV -> {csv_path}")

    # Plot and save confusion matrix
    cm_path = os.path.join(args.out_dir, f"confusion_matrix_pretrained_{args.pretrained}.png")
    cm = plot_confusion_matrix(y_true, y_pred, dataset_classes, cm_path)
    print(f"[SAVED] Confusion matrix image -> {cm_path}")

    # Per-class accuracy (optional)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, cls in enumerate(dataset_classes):
        acc_pct = per_class_acc[i] * 100 if not np.isnan(per_class_acc[i]) else 0.0
        print(f"[CLASS ACC] {cls}: {acc_pct:.2f}%")

    print("[DONE] Evaluation finished.")
