import json
import numpy as np
from collections import defaultdict
from typing import Dict
import matplotlib.pyplot as plt
from pathlib import Path


class DatasetBalancer:
    def __init__(
        self,
        ddscores_path: str,
        n_bins: int = 40,
        max_score: float = 0.8,
    ):
        self.n_bins = n_bins
        self.max_score = max_score
        self.bin_edges = np.linspace(0, max_score, n_bins + 1)

        with open(ddscores_path, "r") as f:
            ddscores_full = json.load(f)

        self.ddscores = {}
        for img_id, score in ddscores_full.items():
            if score <= self.max_score:
                self.ddscores[img_id] = score

        self.total_images = len(self.ddscores)
        self.total_scores = len(self.ddscores)
        self.availability_map = self._create_availability_map()
        self.stats = {
            "initial_stats": self._calculate_initial_stats(),
            "final_stats": {},
            "bin_stats": {},
        }

    def _calculate_initial_stats(self) -> Dict:
        images_per_bin = defaultdict(set)
        for img_id, score in self.ddscores.items():
            bin_idx = np.digitize(score, self.bin_edges) - 1
            if bin_idx < self.n_bins:
                images_per_bin[bin_idx].add(img_id)

        return {
            "total_images": self.total_images,
            "total_scores": self.total_scores,
            "avg_scores_per_image": 1.0,
            "images_per_bin": {
                bin_idx: len(images) for bin_idx, images in images_per_bin.items()
            },
            "score_distribution": {
                f"{self.bin_edges[i]:.2f}-{self.bin_edges[i + 1]:.2f}": len(
                    images_per_bin[i]
                )
                for i in range(self.n_bins)
            },
        }

    def _create_availability_map(self) -> Dict:
        availability = defaultdict(list)
        for img_id, score in self.ddscores.items():
            bin_idx = np.digitize(score, self.bin_edges) - 1
            if bin_idx < self.n_bins:
                availability[bin_idx].append(
                    {"img_id": img_id, "score": score}
                )
        return availability

    def _update_final_stats(self, selected_items: Dict, bin_counts: Dict):
        if not selected_items:
            self.stats["final_stats"] = {
                "images_used": 0,
                "usage_percentage": 0,
                "unused_images": self.total_images,
                "bins_filled": 0,
                "average_images_per_bin": 0,
            }
            self.stats["bin_stats"] = {
                "bin_counts": {},
                "bin_percentages": {},
                "deviation_from_target": {},
                "unfilled_bins": list(range(self.n_bins)),
                "overfilled_bins": [],
                "underfilled_bins": [],
            }
            return

        self.stats["final_stats"] = {
            "images_used": len(selected_items),
            "usage_percentage": (len(selected_items) / self.total_images) * 100,
            "unused_images": self.total_images - len(selected_items),
            "bins_filled": len(bin_counts),
            "average_images_per_bin": len(selected_items) / self.n_bins,
        }

        target_count = len(selected_items) / self.n_bins

        self.stats["bin_stats"] = {
            "bin_counts": dict(bin_counts),
            "bin_percentages": {
                bin_: (count / len(selected_items)) * 100
                for bin_, count in bin_counts.items()
            },
            "deviation_from_target": {
                bin_: count - target_count for bin_, count in bin_counts.items()
            },
            "unfilled_bins": [i for i in range(self.n_bins) if i not in bin_counts],
            "overfilled_bins": [
                bin_ for bin_, count in bin_counts.items() if count > target_count * 1.1
            ],
            "underfilled_bins": [
                bin_ for bin_, count in bin_counts.items() if count < target_count * 0.9
            ],
        }

    def _convert_numpy_types(self, data):
        if isinstance(data, dict):
            return {str(k): self._convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(x) for x in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        else:
            return data

    def create_balanced_dataset(self) -> Dict:
        print("\nInitializing dataset balancer...")
        selected_items = {}
        image_to_bin = {}
        bin_to_images = defaultdict(set)

        # Build maps
        for bin_idx, items in self.availability_map.items():
            for item in items:
                img_id = item["img_id"]
                image_to_bin[img_id] = bin_idx
                bin_to_images[bin_idx].add(img_id)

        # Print initial availability
        print("Initial availability:")
        for bin_idx in range(self.n_bins):
            print(f"Bin {bin_idx}: {len(bin_to_images[bin_idx])} images available")

        # Determine the minimum available count across all bins
        min_images_per_bin = min(len(imgs) for imgs in bin_to_images.values() if len(imgs) > 0)
        print(f"\nTarget images per bin: {min_images_per_bin}")

        bin_counts = defaultdict(int)

        # Randomly sample images from each bin up to the min_images_per_bin count
        for bin_idx in range(self.n_bins):
            available = list(bin_to_images[bin_idx])
            if len(available) >= min_images_per_bin:
                selected_ids = np.random.choice(available, min_images_per_bin, replace=False)
                for img_id in selected_ids:
                    score = self.ddscores[img_id]
                    selected_items[img_id] = {
                        "score": score,
                        "bin": bin_idx,
                    }
                    bin_counts[bin_idx] += 1

        # Final summary
        print("\nFinal distribution:")
        final_counts = [bin_counts[i] for i in range(self.n_bins)]
        print(f"Min: {min(final_counts)}, Max: {max(final_counts)}")
        print(f"Total images selected: {len(selected_items)}")

        self._update_final_stats(selected_items, bin_counts)
        return selected_items

    def print_statistics(self):
        print("\n=== Initial Dataset Statistics ===")
        print(f"Total images: {self.stats['initial_stats']['total_images']}")
        print(f"Total dd scores: {self.stats['initial_stats']['total_scores']}")
        print(f"Average scores per image: 1.00")

        if self.stats["final_stats"]:
            print("\n=== Final Dataset Statistics ===")
            print(f"Images used: {self.stats['final_stats']['images_used']}")
            print(f"Usage percentage: {self.stats['final_stats']['usage_percentage']:.2f}%")
            print(f"Unused images: {self.stats['final_stats']['unused_images']}")
            print(f"Average images per bin: {self.stats['final_stats']['average_images_per_bin']:.2f}")

            print("\n=== Bin Statistics ===")
            print(f"Unfilled bins: {len(self.stats['bin_stats']['unfilled_bins'])}")
            print(f"Overfilled bins: {len(self.stats['bin_stats']['overfilled_bins'])}")
            print(f"Underfilled bins: {len(self.stats['bin_stats']['underfilled_bins'])}")

            print("\nBin Distribution:")
            max_count = max(self.stats["bin_stats"]["bin_counts"].values(), default=1)
            for bin_ in range(self.n_bins):
                count = self.stats["bin_stats"]["bin_counts"].get(bin_, 0)
                bar_length = int((count / max_count) * 50)
                print(
                    f"Bin {bin_:2d} [{self.bin_edges[bin_]:.2f}-{self.bin_edges[bin_ + 1]:.2f}]: "
                    f"{'#' * bar_length} ({count})"
                )


def validate_dataset(json_file, img_file, n_bins, max_score):
    with open(json_file, "r") as f:
        data = json.load(f)

    selected_items = data["selected_items"]
    image_count = defaultdict(int)
    for img_name in selected_items:
        image_count[img_name] += 1

    duplicates = {img: count for img, count in image_count.items() if count > 1}
    if duplicates:
        print("WARNING: Found duplicate images:")
        for img, count in duplicates.items():
            print(f"Image {img} appears {count} times")
    else:
        print("No duplicate images found.")

    bin_edges = np.linspace(0, max_score, n_bins + 1)
    bin_distribution = np.zeros(n_bins, dtype=int)
    incorrect_bins = []

    for img_name, info in selected_items.items():
        score = info["score"]
        assigned_bin = info["bin"]

        correct_bin = np.digitize(score, bin_edges) - 1
        if correct_bin != assigned_bin:
            incorrect_bins.append({
                "image": img_name,
                "score": score,
                "assigned_bin": assigned_bin,
                "correct_bin": correct_bin,
                "bin_range": f"[{bin_edges[assigned_bin]:.3f}, {bin_edges[assigned_bin + 1]:.3f})",
            })

        if 0 <= assigned_bin < n_bins:
            bin_distribution[assigned_bin] += 1

    if incorrect_bins:
        print("\nWARNING: Found incorrect bin assignments:")
        for error in incorrect_bins:
            print(
                f"Image {error['image']} with score {error['score']:.4f} "
                f"is in bin {error['assigned_bin']} (range {error['bin_range']}) "
                f"but should be in bin {error['correct_bin']}"
            )
    else:
        print("\nAll bin assignments are correct.")

    plt.figure(figsize=(15, 7))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, bin_distribution, width=0.018, alpha=0.7)
    plt.xlabel(f"Score Range (0-{max_score})")
    plt.ylabel("Number of Images")
    plt.title(f"Distribution of Images Across Bins (0-{max_score})")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_score)
    plt.xticks(bin_edges[::2], [f"{x:.2f}" for x in bin_edges[::2]], rotation=45)
    plt.tight_layout()
    plt.savefig(img_file)

    return {
        "has_duplicates": bool(duplicates),
        "incorrect_bins": incorrect_bins,
        "distribution": bin_distribution.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


if __name__ == "__main__":
    N_BINS = 30
    MAX_SCORE = 0.8
    ATTEMPT = "01_coco17complete_320p_sr_subsamp_444"
    BASE_DIR = f"ddscores_analysis/mapping/{ATTEMPT}"
    SPLITS = ["train", "val", "test"]

    for split in SPLITS:
        print(f"\n\n{'=' * 50}")
        print(f"Processing {split.upper()} dataset")
        print(f"{'=' * 50}\n")

        output_dir = Path(BASE_DIR) / split
        output_dir.mkdir(parents=True, exist_ok=True)

        ddscores_path = output_dir / "ddscores.json"
        output_json = output_dir / "balanced_dataset.json"
        output_img = output_dir / "balanced_dataset.png"

        if not ddscores_path.exists():
            print(f"Warning: No dd scores found for {split} split at {ddscores_path}")
            continue

        print(f"Loading dd scores from: {ddscores_path}")

        balancer = DatasetBalancer(
            ddscores_path=str(ddscores_path), n_bins=N_BINS, max_score=MAX_SCORE
        )
        selected_items = balancer.create_balanced_dataset()
        balancer.print_statistics()

        with open(output_json, "w") as f:
            json.dump(
                {
                    "selected_items": balancer._convert_numpy_types(selected_items),
                    "statistics": balancer._convert_numpy_types(balancer.stats),
                },
                f,
                indent=2,
            )

        validate_dataset(str(output_json), str(output_img), N_BINS, MAX_SCORE)
