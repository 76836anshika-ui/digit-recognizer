"""Evaluate a trained model: confusion matrix, per-class report, and misclassified samples.

Usage:
    python evaluate.py --weights digit_cnn.pt
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data import get_loaders, MNIST_MEAN, MNIST_STD
from model import DigitCNN
from train import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="digit_cnn.pt")
    args = p.parse_args()

    device = get_device()
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    _, _, test_loader = get_loaders()
    preds, labels, images = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
            images.append(x)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    images = torch.cat(images).squeeze(1).numpy() * MNIST_STD + MNIST_MEAN

    print(classification_report(labels, preds, digits=4))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)

    # Misclassified examples
    wrong = np.where(preds != labels)[0][:25]
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for a in axes.ravel():
        a.axis("off")
    for a, idx in zip(axes.ravel(), wrong):
        a.imshow(images[idx], cmap="gray")
        a.set_title(f"true {labels[idx]} / pred {preds[idx]}", fontsize=8)
    plt.tight_layout()
    plt.savefig("misclassified.png", dpi=120)
    print("Saved confusion_matrix.png and misclassified.png")


if __name__ == "__main__":
    main()
