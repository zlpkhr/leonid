import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optim
import coremltools as ct
from coremltools.optimize.torch.quantization import (
    PostTrainingQuantizerConfig,
    PostTrainingQuantizer,
)


class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = nn.MaxPool2d(kernel_size=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=2)
        self.f5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.c1(x)
        x = F.relu(x)
        x = self.s2(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.s4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.f5(x)
        x = F.relu(x)
        x = self.f6(x)
        x = F.relu(x)
        x = self.f7(x)
        return x


def train_model(model, train_loader, validation_loader, epochs=3):
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs, labels

            optimizer.zero_grad()

            activation = model(inputs)
            loss = F.cross_entropy(activation, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {running_loss/100:.3f}")
                running_loss = 0.0

        model.eval()
        correct = total = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                _, predicted = model(inputs).max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch + 1}, Validation Accuracy: {100. * correct / total:.2f}%")


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_set = datasets.MNIST(
        root="tmp/datasets", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="tmp/datasets", train=False, download=True, transform=transform
    )

    train_set_size = int(len(train_set) * 0.8)
    val_set_size = len(train_set) - train_set_size

    train_set, val_set = data.random_split(train_set, [train_set_size, val_set_size])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    model = Lenet()
    train_model(model, train_loader, val_loader)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs, labels
            _, predicted = model(inputs).max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"Test Accuracy: {100. * correct / total:.2f}%")

    torch.jit.script(model).save("tmp/lenet.pt")

    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": "int8",
            },
        }
    )

    ptq = PostTrainingQuantizer(model, config)
    quantized_model = ptq.compress()

    torch.jit.script(quantized_model).save("tmp/lenet_int8.pt")
