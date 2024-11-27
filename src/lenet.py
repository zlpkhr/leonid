import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1: Convolutional Layer
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        # S2: MaxPooling Layer
        self.s2 = nn.MaxPool2d(kernel_size=2)
        # C3: Convolutional Layer
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        # S4: MaxPooling Layer
        self.s4 = nn.MaxPool2d(kernel_size=2)
        # F5: Fully Connected Layer
        self.f5 = nn.Linear(16 * 5 * 5, 120)
        # F6: Fully Connected Layer
        self.f6 = nn.Linear(120, 84)
        # F7: Output Layer
        self.f7 = nn.Linear(84, 10)
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.s2(x)
        x = self.relu(self.c3(x))
        x = self.s4(x)
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = self.relu(self.f5(x))
        x = self.relu(self.f6(x))
        x = self.f7(x)
        return x


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_model(model, train_loader, val_loader, epochs=3, device=None):
    if device is None:
        device = get_device()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        # Training
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch + 1}, Batch: {i}, Loss: {loss.item():.3f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f"Epoch: {epoch + 1}, Validation Accuracy: {100. * correct / total:.2f}%")


def main():
    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Load and preprocess MNIST dataset
    transform = transforms.Compose(
        [
            transforms.Pad(2),  # Pad to 32x32
            transforms.ToTensor(),
        ]
    )

    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root="tmp/data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="tmp/data", train=False, download=True, transform=transform
    )

    # Create validation split
    train_size = len(train_dataset) - 5000
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, 5000]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize and train model
    model = LeNet5().to(device)
    train_model(model, train_loader, val_loader, epochs=3, device=device)

    # Test evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Test Accuracy: {100. * correct / total:.2f}%")

    # Save model
    torch.save(model.state_dict(), "tmp/lenet5_model.pth")


if __name__ == "__main__":
    main()
