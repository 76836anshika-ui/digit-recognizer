"""Predict the digit in an image file (photo/scan/screenshot of a handwritten digit).

Usage:
    python predict.py my_digit.png
    python predict.py my_digit.png --invert    # use if the image is dark ink on white paper
"""
import argparse

import numpy as np
import torch
from PIL import Image, ImageOps

from data import MNIST_MEAN, MNIST_STD
from model import DigitCNN
from train import get_device


def preprocess(img: Image.Image, invert: bool = False) -> torch.Tensor:
    """Turn any image into a normalized (1, 1, 28, 28) tensor shaped like MNIST:
    white digit on black background, centered in a 28x28 frame with a 4px margin."""
    img = img.convert("L")
    if invert:
        img = ImageOps.invert(img)

    arr = np.array(img, dtype=np.float32)
    # Crop to the digit's bounding box
    ys, xs = np.where(arr > 30)
    if len(xs) == 0:
        raise ValueError("No digit found in the image (is it blank? try --invert).")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Resize longest side to 20px, keep aspect ratio
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = np.array(Image.fromarray(arr.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS),
                     dtype=np.float32)

    # Paste in a 28x28 canvas, centered by center of mass (as MNIST was made)
    canvas = np.zeros((28, 28), dtype=np.float32)
    total = small.sum() or 1.0
    cy = (small.sum(axis=1) * np.arange(new_h)).sum() / total
    cx = (small.sum(axis=0) * np.arange(new_w)).sum() / total
    top = int(np.clip(round(14 - cy), 0, 28 - new_h))
    left = int(np.clip(round(14 - cx), 0, 28 - new_w))
    canvas[top:top + new_h, left:left + new_w] = small

    t = torch.from_numpy(canvas / 255.0)
    t = (t - MNIST_MEAN) / MNIST_STD
    return t.unsqueeze(0).unsqueeze(0)


def load_model(weights="digit_cnn.pt", device=None):
    device = device or get_device()
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model, device


def predict(model, device, img, invert=False):
    x = preprocess(img, invert).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--weights", default="digit_cnn.pt")
    p.add_argument("--invert", action="store_true")
    args = p.parse_args()

    model, device = load_model(args.weights)
    digit, probs = predict(model, device, Image.open(args.image), args.invert)
    print(f"Predicted digit: {digit}  (confidence {probs[digit]:.1%})")
    for d in np.argsort(probs)[::-1][:3]:
        print(f"  {d}: {probs[d]:.1%}")


if __name__ == "__main__":
    main()
