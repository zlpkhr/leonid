import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import seaborn as sns
from datetime import datetime
import pandas as pd


class FaultAnalyzer:
    def __init__(self, log_dir: str = "tmp/fault_injection_logs"):
        self.log_dir = Path(log_dir)
        self.experiments = []
        self.metadata = {}

    def load_results(self, filename: str = None):
        """Load results from a specific file or all files in directory"""
        if filename:
            files = [self.log_dir / filename]
        else:
            files = list(self.log_dir.glob("tmp/fault_injection_*.json"))

        print(f"Found {len(files)} result files")
        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    self.metadata[file_path.stem] = data.get("metadata", {})
                    experiments = data.get("experiments", [])
                    # Filter out None or invalid experiments
                    valid_experiments = [exp for exp in experiments if exp is not None]
                    self.experiments.extend(valid_experiments)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

        print(f"Loaded {len(self.experiments)} valid experiments")

    def calculate_basic_statistics(self) -> Dict:
        """Calculate basic statistics about fault injections"""
        total_exps = len(self.experiments)
        if total_exps == 0:
            print("No experiments to analyze!")
            return {}

        correct_classifications = sum(
            1
            for exp in self.experiments
            if exp["faulty_classification"] == exp["true_label"]
        )

        # Analyze misclassifications
        misclassifications = defaultdict(int)
        digit_failures = defaultdict(int)  # Track failures per digit
        digit_totals = defaultdict(int)  # Track total experiments per digit

        for exp in self.experiments:
            true_label = exp["true_label"]
            faulty_label = exp["faulty_classification"]
            digit_totals[true_label] += 1

            if faulty_label != true_label:
                misclassifications[f"{true_label}->{faulty_label}"] += 1
                digit_failures[true_label] += 1

        # Calculate per-digit vulnerability
        digit_vulnerability = {
            digit: (
                digit_failures[digit] / digit_totals[digit] * 100
                if digit_totals[digit] > 0
                else 0
            )
            for digit in range(10)
        }

        # Layer analysis
        layer_stats = defaultdict(lambda: {"total": 0, "failures": 0})
        for exp in self.experiments:
            layer = exp["layer_name"]
            layer_stats[layer]["total"] += 1
            if exp["faulty_classification"] != exp["true_label"]:
                layer_stats[layer]["failures"] += 1

        # Bit position analysis
        bit_position_stats = defaultdict(lambda: {"total": 0, "failures": 0})
        for exp in self.experiments:
            bit = exp["bit_position"]
            bit_position_stats[bit]["total"] += 1
            if exp["faulty_classification"] != exp["true_label"]:
                bit_position_stats[bit]["failures"] += 1

        # Calculate confidence impacts
        confidence_changes = []
        severe_impacts = []  # Track cases where confidence changed dramatically

        for exp in self.experiments:
            orig_conf = exp["original_probabilities"][exp["true_label"]]
            faulty_conf = exp["faulty_probabilities"][exp["true_label"]]
            change = faulty_conf - orig_conf
            confidence_changes.append(change)

            # Track severe impacts (more than 50% confidence drop)
            if change < -0.5:
                severe_impacts.append(
                    {
                        "true_label": exp["true_label"],
                        "predicted": exp["faulty_classification"],
                        "confidence_drop": change,
                        "layer": exp["layer_name"],
                        "bit_position": exp["bit_position"],
                    }
                )

        return {
            "total_experiments": total_exps,
            "correct_classifications": correct_classifications,
            "accuracy": correct_classifications / total_exps * 100,
            "misclassifications": dict(misclassifications),
            "layer_stats": dict(layer_stats),
            "bit_position_stats": dict(bit_position_stats),
            "digit_vulnerability": digit_vulnerability,
            "confidence_stats": {
                "mean_change": np.mean(confidence_changes),
                "std_change": np.std(confidence_changes),
                "max_decrease": min(confidence_changes),
                "max_increase": max(confidence_changes),
            },
            "severe_impacts": severe_impacts,
        }

    def generate_report(self, output_dir: str = "tmp/fault_analysis"):
        """Generate comprehensive analysis report with visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        stats = self.calculate_basic_statistics()
        if not stats:
            return

        # Save JSON report
        with open(output_dir / "detailed_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)

        # 1. Classification Results Plot
        plt.figure(figsize=(10, 6))
        labels = ["Correct", "Incorrect"]
        values = [
            stats["correct_classifications"],
            stats["total_experiments"] - stats["correct_classifications"],
        ]
        plt.pie(values, labels=labels, autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"])
        plt.title("Classification Results After Fault Injection")
        plt.savefig(output_dir / "classification_results.png")
        plt.close()

        # 2. Layer Impact Plot
        plt.figure(figsize=(12, 6))
        layer_names = list(stats["layer_stats"].keys())
        failure_rates = [
            stats["layer_stats"][l]["failures"] / stats["layer_stats"][l]["total"] * 100
            for l in layer_names
        ]

        plt.bar(layer_names, failure_rates, color="#3498db")
        plt.title("Failure Rate by Layer")
        plt.ylabel("Failure Rate (%)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "layer_impact.png")
        plt.close()

        # 3. Bit Position Impact
        plt.figure(figsize=(15, 6))
        bit_positions = sorted(stats["bit_position_stats"].keys())
        failure_rates = [
            stats["bit_position_stats"][b]["failures"]
            / stats["bit_position_stats"][b]["total"]
            * 100
            for b in bit_positions
        ]

        plt.bar(bit_positions, failure_rates, color="#9b59b6")
        plt.title("Failure Rate by Bit Position")
        plt.xlabel("Bit Position")
        plt.ylabel("Failure Rate (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "bit_position_impact.png")
        plt.close()

        # 4. Digit Vulnerability Plot
        plt.figure(figsize=(12, 6))
        digits = range(10)
        vulnerabilities = [stats["digit_vulnerability"][d] for d in digits]

        plt.bar(digits, vulnerabilities, color="#e67e22")
        plt.title("Vulnerability by Digit")
        plt.xlabel("Digit")
        plt.ylabel("Failure Rate (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "digit_vulnerability.png")
        plt.close()

        # Generate text report
        report_path = output_dir / "analysis_report.txt"
        with open(report_path, "w") as f:
            f.write("Fault Injection Analysis Report\n")
            f.write("==============================\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Experiments: {stats['total_experiments']}\n")
            f.write(f"Overall Accuracy: {stats['accuracy']:.2f}%\n\n")

            f.write("Layer-wise Statistics:\n")
            for layer, data in stats["layer_stats"].items():
                failure_rate = data["failures"] / data["total"] * 100
                f.write(f"{layer}: {failure_rate:.2f}% failure rate ")
                f.write(f"({data['failures']}/{data['total']} experiments)\n")

            f.write("\nDigit Vulnerability:\n")
            for digit, vuln in stats["digit_vulnerability"].items():
                f.write(f"Digit {digit}: {vuln:.2f}% failure rate\n")

            f.write("\nTop 10 Most Common Misclassifications:\n")
            sorted_misclass = sorted(
                stats["misclassifications"].items(), key=lambda x: x[1], reverse=True
            )[:10]
            for pattern, count in sorted_misclass:
                f.write(f"{pattern}: {count} occurrences\n")

            f.write("\nSevere Impact Cases (>50% confidence drop):\n")
            for case in sorted(
                stats["severe_impacts"], key=lambda x: x["confidence_drop"]
            )[:10]:
                f.write(f"Digit {case['true_label']} -> {case['predicted']}, ")
                f.write(f"Confidence drop: {case['confidence_drop']*100:.1f}%, ")
                f.write(f"Layer: {case['layer']}, Bit: {case['bit_position']}\n")

        print(f"Analysis report and visualizations saved to {output_dir}")


def main():
    # Initialize analyzer
    analyzer = FaultAnalyzer()

    # Load results
    analyzer.load_results()

    # Generate comprehensive report
    analyzer.generate_report()

    # Print key findings
    stats = analyzer.calculate_basic_statistics()
    if stats:
        print(f"\nKey Findings:")
        print(f"Total experiments: {stats['total_experiments']}")
        print(f"Overall accuracy: {stats['accuracy']:.2f}%")
        print("\nMost vulnerable layers:")
        for layer, data in stats["layer_stats"].items():
            failure_rate = data["failures"] / data["total"] * 100
            print(f"{layer}: {failure_rate:.2f}% failure rate")

        print("\nMost vulnerable digits:")
        for digit, vuln in sorted(
            stats["digit_vulnerability"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"Digit {digit}: {vuln:.2f}% failure rate")


if __name__ == "__main__":
    main()
