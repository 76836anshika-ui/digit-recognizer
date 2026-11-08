"""Train the CNN on MNIST.

Usage:
    python train.py                  # defaults: 12 epochs
    python train.py --epochs 20 --lr 0.001
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import get_loaders
from model import DigitCNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            count += y.size(0)
    return total_loss / count, correct / count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="digit_cnn.pt")
    args = p.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_loaders(args.batch_size)
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), args.out)

        print(f"Epoch {epoch:2d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.1f}s")

    # Final test using the best checkpoint
    model.load_state_dict(torch.load(args.out, map_location=device))
    _, test_acc = run_epoch(model, test_loader, criterion, device)
    print(f"\nBest val acc: {best_val:.4f} | Test accuracy: {test_acc:.4f}")

    # Training curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    print("Saved model to", args.out, "and curves to training_curves.png")


if __name__ == "__main__":
    main()
