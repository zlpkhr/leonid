import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

# Import LeNet5 class from your model file
from lenet import LeNet5


def evaluate_model(model, device="cpu"):
    """Quick evaluation of model accuracy"""
    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
        ]
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


class QuantizedLeNet5(nn.Module):
    def __init__(self, original_model):
        super(QuantizedLeNet5, self).__init__()
        self.min_val = -128
        self.max_val = 127

        # Store the original model's layers
        self.layers = {
            "c1": original_model.c1,
            "c3": original_model.c3,
            "f5": original_model.f5,
            "f6": original_model.f6,
            "f7": original_model.f7,
        }

        # Non-quantized layers
        self.s2 = original_model.s2
        self.s4 = original_model.s4
        self.relu = original_model.relu

        # Store quantization parameters
        self.scales = {}
        self.zero_points = {}
        self.quantized_weights = {}
        self.quantized_biases = {}

        # Quantize weights and store parameters
        self.quantize_weights()

    def quantize_layer(self, name, layer):
        # Quantize weights
        weight = layer.weight.data
        w_min = weight.min().item()
        w_max = weight.max().item()

        # Calculate scale and zero point
        w_scale = (w_max - w_min) / (self.max_val - self.min_val)
        w_zero_point = self.min_val - round(w_min / w_scale)

        # Quantize to int8
        w_quant = torch.clamp(
            torch.round(weight / w_scale) + w_zero_point, self.min_val, self.max_val
        ).to(torch.int8)

        # Store quantization parameters
        self.scales[f"{name}_w"] = w_scale
        self.zero_points[f"{name}_w"] = w_zero_point
        self.quantized_weights[name] = w_quant

        # Quantize bias if present
        if layer.bias is not None:
            bias = layer.bias.data
            b_min = bias.min().item()
            b_max = bias.max().item()

            b_scale = (b_max - b_min) / (self.max_val - self.min_val)
            b_zero_point = self.min_val - round(b_min / b_scale)

            b_quant = torch.clamp(
                torch.round(bias / b_scale) + b_zero_point, self.min_val, self.max_val
            ).to(torch.int8)

            self.scales[f"{name}_b"] = b_scale
            self.zero_points[f"{name}_b"] = b_zero_point
            self.quantized_biases[name] = b_quant

    def quantize_weights(self):
        for name, layer in self.layers.items():
            self.quantize_layer(name, layer)

    def dequantize_layer(self, name):
        # Dequantize weights
        w_scale = self.scales[f"{name}_w"]
        w_zero_point = self.zero_points[f"{name}_w"]
        weight = (self.quantized_weights[name].float() - w_zero_point) * w_scale

        # Dequantize bias if present
        bias = None
        if name in self.quantized_biases:
            b_scale = self.scales[f"{name}_b"]
            b_zero_point = self.zero_points[f"{name}_b"]
            bias = (self.quantized_biases[name].float() - b_zero_point) * b_scale

        return weight, bias

    def forward(self, x):
        # C1 layer
        w1, b1 = self.dequantize_layer("c1")
        x = torch.nn.functional.conv2d(
            x,
            w1,
            b1,
            stride=self.layers["c1"].stride,
            padding=self.layers["c1"].padding,
        )
        x = self.relu(x)
        x = self.s2(x)

        # C3 layer
        w3, b3 = self.dequantize_layer("c3")
        x = torch.nn.functional.conv2d(
            x,
            w3,
            b3,
            stride=self.layers["c3"].stride,
            padding=self.layers["c3"].padding,
        )
        x = self.relu(x)
        x = self.s4(x)

        x = x.view(-1, 16 * 5 * 5)

        # F5 layer
        w5, b5 = self.dequantize_layer("f5")
        x = torch.nn.functional.linear(x, w5, b5)
        x = self.relu(x)

        # F6 layer
        w6, b6 = self.dequantize_layer("f6")
        x = torch.nn.functional.linear(x, w6, b6)
        x = self.relu(x)

        # F7 layer
        w7, b7 = self.dequantize_layer("f7")
        x = torch.nn.functional.linear(x, w7, b7)

        return x

    def save_quantized(self, path):
        """Save quantized weights and parameters in int8 format"""
        save_dict = {
            "scales": self.scales,
            "zero_points": self.zero_points,
            "quantized_weights": self.quantized_weights,
            "quantized_biases": self.quantized_biases,
        }
        torch.save(save_dict, path)

    @staticmethod
    def load_quantized(path, original_model):
        """Load quantized weights and parameters"""
        quantized_model = QuantizedLeNet5(original_model)
        saved_dict = torch.load(path)
        quantized_model.scales = saved_dict["scales"]
        quantized_model.zero_points = saved_dict["zero_points"]
        quantized_model.quantized_weights = saved_dict["quantized_weights"]
        quantized_model.quantized_biases = saved_dict["quantized_biases"]
        return quantized_model


def main():
    # Load the trained model
    model_path = Path("lenet5_model.pth")
    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Please train the model first.")

    # Initialize model and load weights
    model = LeNet5()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Print initial model size
    torch.save(model.state_dict(), "temp_float.pth")
    float_size = Path("temp_float.pth").stat().st_size / (1024 * 1024)  # Size in MB
    print(f"Float Model Size: {float_size:.2f} MB")

    # Evaluate float model
    float_accuracy = evaluate_model(model)
    print(f"Float Model Accuracy: {float_accuracy:.2f}%")

    # Create and evaluate quantized model
    print("Applying int8 quantization...")
    quantized_model = QuantizedLeNet5(model)

    # Save quantized model with int8 weights
    quantized_model.save_quantized("lenet5_int8.pth")
    quantized_size = Path("lenet5_int8.pth").stat().st_size / (
        1024 * 1024
    )  # Size in MB

    # Evaluate quantized model
    quantized_accuracy = evaluate_model(quantized_model)

    # Print results
    print("\nResults:")
    print(f"Float Model Size: {float_size:.2f} MB")
    print(f"Quantized Model Size: {quantized_size:.2f} MB")
    print(f"Size Reduction: {(1 - quantized_size/float_size)*100:.1f}%")
    print(f"Float Model Accuracy: {float_accuracy:.2f}%")
    print(f"Quantized Model Accuracy: {quantized_accuracy:.2f}%")
    print(f"Accuracy Drop: {float_accuracy - quantized_accuracy:.2f}%")

    # Clean up temporary file
    Path("tmp/temp_float.pth").unlink()


if __name__ == "__main__":
    main()
