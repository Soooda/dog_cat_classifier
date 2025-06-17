import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.feature = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
        )
        # Dummy input to calculate flattened size
        dummy_input = torch.zeros(1, 3, width, height)
        x = self.feature(dummy_input)
        flat_features = x.view(1, -1).size(1)
        self.classifier = nn.Sequential(
            nn.Linear(flat_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.classifier(self.feature(x))
        return x

if __name__ == '__main__':
    model = CNN(128, 128)
    dummy_input = torch.ones(12, 3, 128, 128)
    ret = model(dummy_input)
    print(ret)
