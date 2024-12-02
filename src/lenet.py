import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.s2(x)
        x = self.relu(self.c3(x))
        x = self.s4(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.f5(x))
        x = self.relu(self.f6(x))
        return self.f7(x)


def train_model(model, train_loader, validation_loader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs, labels

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
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
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="tmp/datasets/mnist", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="tmp/datasets/mnist", train=False, download=True, transform=transform
    )

    train_size = len(train_dataset) - 5000
    train_dataset, validation_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, 5000]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = Lenet()
    train_model(model, train_loader, validation_loader)

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
