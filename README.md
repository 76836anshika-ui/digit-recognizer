# Handwritten Digit Recognizer (MNIST + CNN)

A Convolutional Neural Network, built with PyTorch, that recognizes handwritten digits (0-9)
from the MNIST dataset (60,000 training / 10,000 test images, 28x28 grayscale).
It reaches roughly **99.5% test accuracy** and includes a draw-your-own-digit app.

## Project structure

| File | Purpose |
|---|---|
| `model.py` | CNN architecture (`DigitCNN`) |
| `data.py` | MNIST download, augmentation, train/val/test loaders |
| `train.py` | Training loop, best-checkpoint saving, loss/accuracy curves |
| `evaluate.py` | Classification report, confusion matrix, misclassified examples |
| `predict.py` | Predict from an image file; MNIST-style preprocessing |
| `app.py` | Tkinter canvas: draw a digit, see live predictions |
| `requirements.txt` | Dependencies |

## Quick start

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train.py            # downloads MNIST, trains ~12 epochs, saves digit_cnn.pt
python evaluate.py         # confusion_matrix.png, misclassified.png
python app.py              # draw digits with the mouse
python predict.py digit.png          # white digit on black background
python predict.py scan.png --invert  # dark ink on white paper
```

Training takes a few minutes on CPU and under a minute per epoch on a GPU.

## How it works

**Architecture** (about 0.5M parameters):

```
Input 1x28x28
 -> [Conv3x3(32) -> BN -> ReLU] x2 -> MaxPool      32x14x14
 -> [Conv3x3(64) -> BN -> ReLU] x2 -> MaxPool      64x7x7
 -> Flatten -> Dropout -> Linear(128) -> ReLU -> Dropout -> Linear(10)
```

**Key ideas to learn from this project**

- **Convolutions** learn small filters (edges, strokes, loops) shared across the image, so the
  network needs far fewer parameters than a fully connected one.
- **Pooling** halves the spatial size, adding tolerance to small shifts.
- **Batch normalization + dropout** stabilize training and reduce overfitting.
- **Data augmentation** (small rotations, shifts, zoom) makes the model robust to real handwriting.
- **Train / validation / test split**: the best checkpoint is chosen on validation data,
  and the test set is used only once at the end.
- **Preprocessing matters**: `predict.py` crops, scales to 20px, and centers the digit by center
  of mass, exactly how MNIST was built. Skipping this is the most common reason a model that
  scores 99% on MNIST fails on your own drawings.

## Experiments to try

1. Remove augmentation and compare validation accuracy.
2. Remove BatchNorm or Dropout and watch the training curves.
3. Change filter counts or add a third conv block.
4. Compare with a plain MLP (flatten -> 256 -> 10) to see what convolutions buy you.
5. Visualize the first-layer filters or feature maps.
6. Try Fashion-MNIST by swapping `datasets.MNIST` for `datasets.FashionMNIST` in `data.py`.
7. Export to ONNX (`torch.onnx.export`) and run it in the browser.

## Typical results

| Metric | Value |
|---|---|
| Validation accuracy | ~99.4-99.6% |
| Test accuracy | ~99.4-99.6% |
| Common confusions | 4 vs 9, 7 vs 1, 3 vs 5 |
