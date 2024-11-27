import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import json
from pathlib import Path
import time
from copy import deepcopy


@dataclass
class FaultInjectionRecord:
    """Record of a single fault injection"""

    # Identification
    experiment_id: str
    timestamp: float
    image_id: int  # MNIST identifier
    true_label: int  # Correct digit label

    # Injection Location
    layer_name: str  # 'c1', 'f5', etc.
    tensor_shape: Tuple[int, ...]

    # Bit-level Information
    word_index: Tuple[int, ...]  # Location in tensor
    bit_position: int  # Position within word (0-7 for int8)
    original_value: int  # Original quantized value
    flipped_value: int  # Flipped quantized value

    # Original Results
    original_classification: int
    original_probabilities: List[float]  # Probabilities for all digits [0-9]

    # Faulty Results
    faulty_classification: int
    faulty_probabilities: List[float]  # Probabilities for all digits [0-9]

    def to_dict(self) -> dict:
        """Convert record to dictionary for JSON serialization"""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "image_id": int(self.image_id),
            "true_label": int(self.true_label),
            "layer_name": self.layer_name,
            "tensor_shape": list(self.tensor_shape),
            "word_index": list(self.word_index),
            "bit_position": int(self.bit_position),
            "original_value": int(self.original_value),
            "flipped_value": int(self.flipped_value),
            "original_classification": int(self.original_classification),
            "original_probabilities": [float(p) for p in self.original_probabilities],
            "faulty_classification": int(self.faulty_classification),
            "faulty_probabilities": [float(p) for p in self.faulty_probabilities],
        }


class FaultInjectionLogger:
    """Logger for fault injection experiments"""

    def __init__(self, output_dir: str = "fault_injection_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.current_log = []

    def log_injection(self, record: FaultInjectionRecord):
        """Log a fault injection record"""
        self.current_log.append(record.to_dict())

    def save_logs(self, filename: str):
        """Save current logs to file"""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "timestamp": time.time(),
                        "total_experiments": len(self.current_log),
                    },
                    "experiments": self.current_log,
                },
                f,
                indent=2,
            )
        print(f"Saved {len(self.current_log)} experiments to {output_path}")
        self.current_log = []  # Clear after saving


class FaultInjectionLeNet5(nn.Module):
    def __init__(self, quantized_model):
        super(FaultInjectionLeNet5, self).__init__()
        self.quantized_model = quantized_model
        self.logger = FaultInjectionLogger()
        self.current_image_id = None
        self.current_true_label = None
        self.fault_active = False
        self.current_fault = None

        # Store original parameters for restoration
        self.original_weights = {}
        for name, tensor_dict in self.quantized_model.quantized_weights.items():
            self.original_weights[name] = tensor_dict.clone()

    @staticmethod
    def flip_bit(value: int, bit_position: int) -> int:
        """Flip a specific bit in an 8-bit integer value"""
        if not 0 <= bit_position < 8:
            raise ValueError("Bit position must be between 0 and 7 for int8")
        return value ^ (1 << bit_position)

    def inject_fault_tensor(
        self, tensor: torch.Tensor, word_index: Tuple[int, ...], bit_position: int
    ) -> Tuple[int, int, torch.Tensor]:
        """Inject a fault into a specific position in a tensor"""
        faulty_tensor = tensor.clone()

        # Get original int8 value
        original_value = tensor[word_index].item()
        if not isinstance(original_value, (int, np.int8, np.int32, np.int64)):
            raise ValueError(f"Expected integer value, got {type(original_value)}")

        # Flip bit and ensure result stays in int8 range
        flipped_value = self.flip_bit(original_value, bit_position)
        flipped_value = max(-128, min(127, flipped_value))  # Clamp to int8 range

        # Update tensor
        faulty_tensor[word_index] = flipped_value

        return original_value, flipped_value, faulty_tensor

    def inject_weight_fault(
        self, layer_name: str, word_index: Tuple[int, ...], bit_position: int
    ) -> Tuple[int, int]:
        """Inject fault into quantized weights of specified layer"""
        if layer_name not in self.quantized_model.quantized_weights:
            raise ValueError(f"Invalid layer name: {layer_name}")

        weights = self.quantized_model.quantized_weights[layer_name]
        original_value, flipped_value, faulty_weights = self.inject_fault_tensor(
            weights, word_index, bit_position
        )

        # Update the weights in the quantized model
        self.quantized_model.quantized_weights[layer_name] = faulty_weights

        return original_value, flipped_value

    def prepare_fault_injection(
        self,
        image_id: int,
        true_label: int,
        layer_name: str,
        word_index: Tuple[int, ...],
        bit_position: int,
    ):
        """Prepare for fault injection"""
        self.current_image_id = image_id
        self.current_true_label = true_label
        self.fault_active = True
        self.current_fault = {
            "layer_name": layer_name,
            "word_index": word_index,
            "bit_position": bit_position,
        }

    def restore_original_weights(self):
        """Restore original weights after fault injection"""
        for name, weights in self.original_weights.items():
            self.quantized_model.quantized_weights[name] = weights.clone()

    def clear_fault_injection(self):
        """Clear current fault injection and restore weights"""
        self.fault_active = False
        self.current_fault = None
        self.restore_original_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fault_active and self.current_fault:
            # Get original output and probabilities
            with torch.no_grad():
                orig_output = self.quantized_model(x)
                orig_probs = torch.nn.functional.softmax(orig_output, dim=1)

                # Inject fault into weights
                orig_val, flipped_val = self.inject_weight_fault(
                    self.current_fault["layer_name"],
                    self.current_fault["word_index"],
                    self.current_fault["bit_position"],
                )

                # Get faulty output and probabilities
                faulty_output = self.quantized_model(x)
                faulty_probs = torch.nn.functional.softmax(faulty_output, dim=1)

                # Record injection details
                record = FaultInjectionRecord(
                    experiment_id=f"inj_{time.time()}",
                    timestamp=time.time(),
                    image_id=self.current_image_id,
                    true_label=self.current_true_label,
                    layer_name=self.current_fault["layer_name"],
                    tensor_shape=self.quantized_model.quantized_weights[
                        self.current_fault["layer_name"]
                    ].shape,
                    word_index=self.current_fault["word_index"],
                    bit_position=self.current_fault["bit_position"],
                    original_value=orig_val,
                    flipped_value=flipped_val,
                    original_classification=orig_probs.argmax(1).item(),
                    original_probabilities=orig_probs[0].tolist(),
                    faulty_classification=faulty_probs.argmax(1).item(),
                    faulty_probabilities=faulty_probs[0].tolist(),
                )

                self.logger.log_injection(record)
                return faulty_output

        return self.quantized_model(x)


def random_fault_parameters(model: nn.Module, layer_name: str) -> Tuple:
    """Generate random fault injection parameters for a given layer"""
    tensor = model.quantized_weights[layer_name]

    # Generate random indices for each dimension
    word_index = tuple(torch.randint(0, dim, (1,)).item() for dim in tensor.shape)

    # Random bit position (0-7 for int8)
    bit_position = torch.randint(0, 8, (1,)).item()

    return word_index, bit_position
