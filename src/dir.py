import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns
from collections import defaultdict


def get_bit_flip_direction(
    original_value: int, _flipped_value: int, bit_position: int
) -> str:
    """
    Determine if the bit flip was 0->1 or 1->0 just from original value and bit position
    """
    # Convert to unsigned 8-bit
    original_value = original_value & 0xFF

    # Get the bit at position
    original_bit = (original_value >> bit_position) & 1

    return "0->1" if original_bit == 0 else "1->0"


def analyze_bit_flips(experiments: List[Dict]) -> Dict:
    """Analyze bit flip directions and their impacts"""
    results = {
        "total_flips": defaultdict(int),
        "failures": defaultdict(int),
        "layer_impacts": defaultdict(lambda: defaultdict(int)),
        "bit_position_impacts": defaultdict(lambda: defaultdict(int)),
        "original_values": [],
        "flipped_values": [],
        "flip_directions": [],
        "success_rate": defaultdict(list),
        "detailed_flips": [],
    }

    for exp in experiments:
        direction = get_bit_flip_direction(
            exp["original_value"], exp["flipped_value"], exp["bit_position"]
        )

        # Record flip details
        flip_detail = {
            "original_value": exp["original_value"],
            "original_binary": format(exp["original_value"] & 0xFF, "08b"),
            "flipped_value": exp["flipped_value"],
            "flipped_binary": format(exp["flipped_value"] & 0xFF, "08b"),
            "bit_position": exp["bit_position"],
            "direction": direction,
            "layer": exp["layer_name"],
            "success": exp["faulty_classification"] == exp["true_label"],
            "original_confidence": max(exp["original_probabilities"]),
            "faulty_confidence": max(exp["faulty_probabilities"]),
            "confidence_change": max(exp["faulty_probabilities"])
            - max(exp["original_probabilities"]),
            "true_label": exp["true_label"],
            "faulty_classification": exp["faulty_classification"],
        }
        results["detailed_flips"].append(flip_detail)

        # Count total flips by direction
        results["total_flips"][direction] += 1

        # Track successes/failures
        if exp["faulty_classification"] != exp["true_label"]:
            results["failures"][direction] += 1

        # Track layer-specific impacts
        results["layer_impacts"][exp["layer_name"]][direction] += 1

        # Track bit position impacts
        results["bit_position_impacts"][exp["bit_position"]][direction] += 1

        # Track values
        results["original_values"].append(exp["original_value"])
        results["flipped_values"].append(exp["flipped_value"])
        results["flip_directions"].append(direction)

        # Track success rate
        is_success = exp["faulty_classification"] == exp["true_label"]
        results["success_rate"][direction].append(is_success)

    return results


def generate_bit_flip_report(results: Dict, output_dir: str = "tmp/bit_flip_analysis"):
    """Generate comprehensive report on bit flip analysis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save detailed results
    with open(output_dir / "bit_flip_details.json", "w") as f:
        json.dump(results["detailed_flips"], f, indent=2)

    # 1. Overall Direction Distribution
    plt.figure(figsize=(10, 6))
    directions = list(results["total_flips"].keys())
    counts = [results["total_flips"][d] for d in directions]
    plt.bar(directions, counts)
    plt.title("Distribution of Bit Flip Directions")
    plt.ylabel("Count")
    for i, v in enumerate(counts):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "flip_direction_distribution.png")
    plt.close()

    # 2. Success Rate by Direction
    plt.figure(figsize=(10, 6))
    success_rates = {
        d: (1 - sum(v) / len(v)) * 100 for d, v in results["success_rate"].items()
    }
    plt.bar(success_rates.keys(), success_rates.values())
    plt.title("Success Rate by Flip Direction")
    plt.ylabel("Success Rate (%)")
    for i, (k, v) in enumerate(success_rates.items()):
        plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_by_direction.png")
    plt.close()

    # 3. Layer Impact by Direction
    plt.figure(figsize=(12, 6))
    layers = list(results["layer_impacts"].keys())
    x = np.arange(len(layers))
    width = 0.35

    zero_to_one = [results["layer_impacts"][layer]["0->1"] for layer in layers]
    one_to_zero = [results["layer_impacts"][layer]["1->0"] for layer in layers]

    plt.bar(x - width / 2, zero_to_one, width, label="0->1")
    plt.bar(x + width / 2, one_to_zero, width, label="1->0")

    plt.xlabel("Layer")
    plt.ylabel("Count")
    plt.title("Layer Impact by Flip Direction")
    plt.xticks(x, layers)
    plt.legend()

    # Add value labels
    for i, v in enumerate(zero_to_one):
        plt.text(i - width / 2, v, str(v), ha="center", va="bottom")
    for i, v in enumerate(one_to_zero):
        plt.text(i + width / 2, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_dir / "layer_impact_by_direction.png")
    plt.close()

    # 4. Bit Position Impact
    plt.figure(figsize=(12, 6))
    bit_positions = sorted(results["bit_position_impacts"].keys())
    x = np.arange(len(bit_positions))

    zero_to_one = [
        results["bit_position_impacts"][pos]["0->1"] for pos in bit_positions
    ]
    one_to_zero = [
        results["bit_position_impacts"][pos]["1->0"] for pos in bit_positions
    ]

    plt.bar(x - width / 2, zero_to_one, width, label="0->1")
    plt.bar(x + width / 2, one_to_zero, width, label="1->0")

    plt.xlabel("Bit Position")
    plt.ylabel("Count")
    plt.title("Bit Position Impact by Flip Direction")
    plt.xticks(x, bit_positions)
    plt.legend()

    # Add value labels
    for i, v in enumerate(zero_to_one):
        plt.text(i - width / 2, v, str(v), ha="center", va="bottom")
    for i, v in enumerate(one_to_zero):
        plt.text(i + width / 2, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_dir / "bit_position_impact.png")
    plt.close()

    # Generate text report
    with open(output_dir / "bit_flip_analysis.txt", "w") as f:
        f.write("Bit Flip Analysis Report\n")
        f.write("=======================\n\n")

        f.write("Overall Statistics:\n")
        total_flips = sum(results["total_flips"].values())
        f.write(f"Total number of bit flips analyzed: {total_flips}\n\n")

        for direction, count in results["total_flips"].items():
            failures = results["failures"][direction]
            success_rate = (count - failures) / count * 100 if count > 0 else 0
            f.write(f"{direction}:\n")
            f.write(
                f"  Total flips: {count} ({count/total_flips*100:.1f}% of all flips)\n"
            )
            f.write(f"  Failures: {failures}\n")
            f.write(f"  Success rate: {success_rate:.2f}%\n\n")

        f.write("\nLayer-wise Impact:\n")
        for layer, directions in results["layer_impacts"].items():
            f.write(f"\n{layer}:\n")
            layer_total = sum(directions.values())
            for direction, count in directions.items():
                f.write(
                    f"  {direction}: {count} flips ({count/layer_total*100:.1f}%)\n"
                )

        f.write("\nBit Position Impact:\n")
        for pos in sorted(results["bit_position_impacts"].keys()):
            f.write(f"\nBit {pos}:\n")
            pos_total = sum(results["bit_position_impacts"][pos].values())
            for direction, count in results["bit_position_impacts"][pos].items():
                f.write(f"  {direction}: {count} flips ({count/pos_total*100:.1f}%)\n")

        f.write("\nConfidence Impact:\n")
        for direction in ["0->1", "1->0"]:
            direction_flips = [
                flip
                for flip in results["detailed_flips"]
                if flip["direction"] == direction
            ]
            avg_conf_change = np.mean(
                [flip["confidence_change"] for flip in direction_flips]
            )
            max_conf_drop = min([flip["confidence_change"] for flip in direction_flips])
            f.write(f"\n{direction}:\n")
            f.write(f"  Average confidence change: {avg_conf_change:.4f}\n")
            f.write(f"  Maximum confidence drop: {max_conf_drop:.4f}\n")


def main():
    # Load existing results
    logs_dir = Path("tmp/fault_injection_logs")
    results_files = list(logs_dir.glob("fault_injection_*.json"))

    if not results_files:
        print("No fault injection results found!")
        return

    all_experiments = []
    for file_path in results_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            if "experiments" in data:
                all_experiments.extend(data["experiments"])
            else:
                print(f"Warning: No experiments found in {file_path}")

    print(f"Loaded {len(all_experiments)} experiments")

    # Analyze bit flips
    results = analyze_bit_flips(all_experiments)

    # Generate report
    generate_bit_flip_report(results)

    # Print summary
    print("\nBit Flip Analysis Summary:")
    print("------------------------")
    total_flips = sum(results["total_flips"].values())
    print(f"Total bit flips analyzed: {total_flips}")

    for direction, count in results["total_flips"].items():
        failures = results["failures"][direction]
        success_rate = (count - failures) / count * 100 if count > 0 else 0
        print(f"\n{direction}:")
        print(f"  Total flips: {count} ({count/total_flips*100:.1f}% of all flips)")
        print(f"  Failures: {failures}")
        print(f"  Success rate: {success_rate:.2f}%")


if __name__ == "__main__":
    main()
