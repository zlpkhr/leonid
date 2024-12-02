import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import random
import time
import multiprocessing as mp
from multiprocessing import Pool, Manager
import numpy as np
from typing import Dict, List, Tuple
import os
import json

# Import necessary classes
from lenet import Lenet
from lenet_int8 import QuantizedLeNet5
from fault_injection import FaultInjectionLeNet5, random_fault_parameters


def load_test_data(batch_size=1):
    """Load MNIST test dataset"""
    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
        ]
    )

    test_dataset = torchvision.datasets.MNIST(
        root="tmp/data", train=False, download=True, transform=transform
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def run_single_experiment(args: Tuple) -> Dict:
    """Run a single fault injection experiment"""
    test_loader, model_path, layer_names, experiment_id = args

    # Load model in each process to avoid sharing issues
    original_model = Lenet()
    saved_dict = torch.load(model_path, weights_only=True)
    quantized_model = QuantizedLeNet5(original_model)
    quantized_model.scales = saved_dict["scales"]
    quantized_model.zero_points = saved_dict["zero_points"]
    quantized_model.quantized_weights = saved_dict["quantized_weights"]
    quantized_model.quantized_biases = saved_dict["quantized_biases"]

    fault_model = FaultInjectionLeNet5(quantized_model)

    # Randomly select image
    total_images = len(test_loader)
    random_idx = random.randint(0, total_images - 1)

    # Get the specific image
    data_iter = iter(test_loader)
    for _ in range(random_idx):
        next(data_iter)
    inputs, labels = next(data_iter)

    # Randomly select layer and get fault parameters
    layer_name = random.choice(layer_names)
    word_index, bit_position = random_fault_parameters(
        fault_model.quantized_model, layer_name
    )

    # Prepare and execute fault injection
    fault_model.prepare_fault_injection(
        image_id=random_idx,
        true_label=labels.item(),
        layer_name=layer_name,
        word_index=word_index,
        bit_position=bit_position,
    )

    # Run inference with fault
    _ = fault_model(inputs)

    # Get the experiment record
    record = fault_model.logger.current_log[0]

    # Clear fault injection
    fault_model.clear_fault_injection()

    return record


def run_parallel_fault_injection_campaign(
    model_path: str,
    num_experiments: int = 100,
    layer_names: list = ["c1", "c3", "f5", "f6", "f7"],
    save_interval: int = 50,
    num_processes: int = None,
):
    """Run a parallel fault injection campaign"""
    if num_processes is None:
        num_processes = mp.cpu_count() - 1  # Leave one CPU free

    test_loader = load_test_data(batch_size=1)
    print(
        f"Starting parallel fault injection campaign with {num_experiments} experiments on {num_processes} processes..."
    )

    start_time = time.time()

    # Prepare arguments for each experiment
    args = [(test_loader, model_path, layer_names, i) for i in range(num_experiments)]

    # Create process pool and run experiments
    results = []
    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_experiment, args)):
            results.append(result)

            # Save intermediate results
            if (i + 1) % save_interval == 0:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_results(results, f"fault_injection_batch_{timestamp}.json")

            # Print progress
            if (i + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_exp = elapsed_time / (i + 1)
                remaining_exps = num_experiments - (i + 1)
                estimated_time = remaining_exps * avg_time_per_exp

                print(f"Completed {i + 1}/{num_experiments} experiments")
                print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")

    # Save final results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_results(results, f"fault_injection_final_{timestamp}.json")

    print(f"Completed all experiments in {(time.time() - start_time)/60:.1f} minutes")


def save_results(results: List[Dict], filename: str):
    """Save results to a JSON file"""
    output_dir = Path("tmp/fault_injection_logs")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / filename, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": time.time(),
                    "total_experiments": len(results),
                },
                "experiments": results,
            },
            f,
            indent=2,
        )


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Check for model file
    model_path = Path("tmp/lenet5_int8.pth")
    if not model_path.exists():
        raise FileNotFoundError("Quantized model file not found!")

    # Run parallel campaign
    run_parallel_fault_injection_campaign(
        model_path=str(model_path),
        num_experiments=100,  # Adjust as needed
        save_interval=50,
        num_processes=None,  # Will use CPU count - 1
    )


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
