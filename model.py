"""CNN architecture for MNIST digit recognition."""
import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    """
    Input : (N, 1, 28, 28) normalized grayscale image
    Output: (N, 10) raw class scores (logits)

    Conv(1->32) -> ReLU -> Conv(32->32) -> ReLU -> MaxPool      28x28 -> 14x14
    Conv(32->64) -> ReLU -> Conv(64->64) -> ReLU -> MaxPool     14x14 -> 7x7
    Flatten -> Dropout -> Linear(3136->128) -> ReLU -> Dropout -> Linear(128->10)
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


if __name__ == "__main__":
    net = DigitCNN()
    print(net)
    print("Parameters:", sum(p.numel() for p in net.parameters()))
    print("Output shape:", net(torch.randn(2, 1, 28, 28)).shape)
