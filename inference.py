import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import time

# Import both model classes
from lenet import LeNet5
from lenet_int8 import QuantizedLeNet5


class SafetyError(Exception):
    pass


def safe_load(path, weights_only=True):
    """Safely load model weights"""
    try:
        return torch.load(
            path, weights_only=weights_only, map_location=torch.device("cpu")
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def load_test_data(batch_size=32):
    """Load MNIST test dataset"""
    transform = transforms.Compose(
        [
            transforms.Pad(2),  # Pad to 32x32
            transforms.ToTensor(),
        ]
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    return DataLoader(test_dataset, batch_size=batch_size)


def evaluate_model(model, test_loader, device="cpu"):
    """Evaluate model performance and speed"""
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    batch_sizes = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            current_batch_size = inputs.size(0)

            # Measure inference time
            start_time = time.perf_counter()
            outputs = model(inputs)
            batch_time = time.perf_counter() - start_time

            inference_times.append(batch_time)
            batch_sizes.append(current_batch_size)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    accuracy = 100.0 * correct / total

    # Calculate average times properly accounting for different batch sizes
    total_images = sum(batch_sizes)
    total_time = sum(inference_times)
    avg_time_per_image = total_time / total_images
    avg_time_per_batch = total_time / len(inference_times)

    return {
        "accuracy": accuracy,
        "avg_time_per_batch": avg_time_per_batch,
        "avg_time_per_image": avg_time_per_image,
        "total_images": total_images,
        "total_time": total_time,
    }


def main():
    device = torch.device("cpu")  # You can modify this for GPU if needed

    # Load original model architecture for reference
    original_model = LeNet5()

    # Check for quantized model file
    quantized_path = Path("lenet5_int8.pth")
    if not quantized_path.exists():
        raise FileNotFoundError(
            "Quantized model file not found! Run quantization script first."
        )

    # Load quantized model
    print("Loading quantized model...")
    try:
        saved_dict = safe_load(quantized_path)
        quantized_model = QuantizedLeNet5(original_model)
        quantized_model.scales = saved_dict["scales"]
        quantized_model.zero_points = saved_dict["zero_points"]
        quantized_model.quantized_weights = saved_dict["quantized_weights"]
        quantized_model.quantized_biases = saved_dict["quantized_biases"]
    except Exception as e:
        print(f"Failed to load quantized model: {str(e)}")
        return

    quantized_model = quantized_model.to(device)

    # Load test data
    print("Loading test dataset...")
    test_loader = load_test_data(batch_size=32)

    # Evaluate quantized model
    print("\nRunning inference with quantized model...")
    results = evaluate_model(quantized_model, test_loader, device)

    # Print results
    print("\nQuantized Model Performance:")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(
        f"Average inference time per batch: {results['avg_time_per_batch']*1000:.2f} ms"
    )
    print(
        f"Average inference time per image: {results['avg_time_per_image']*1000:.2f} ms"
    )
    print(f"Total inference time: {results['total_time']:.2f} seconds")
    print(f"Total images processed: {results['total_images']}")

    # Calculate model size
    model_size = quantized_path.stat().st_size / (1024 * 1024)  # Size in MB
    print(f"Model Size: {model_size:.2f} MB")


if __name__ == "__main__":
    main()
