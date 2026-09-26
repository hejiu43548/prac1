"""Run pretrained ResNet on MNIST without training.

The reported raw-index accuracy compares ImageNet class IDs with MNIST digits.
These label spaces differ: this is not meaningful digit-recognition accuracy.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torchvision.datasets import MNIST
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0,
                        help="Number of test images; 0 uses the full test set")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.limit < 0:
        parser.error("batch-size must be positive and limit must be nonnegative")

    mps_available = torch.backends.mps.is_available()
    if args.device == "mps" and not mps_available:
        parser.error("MPS is unavailable; use --device cpu")
    device = torch.device(
        ("mps" if mps_available else "cpu")
        if args.device == "auto" else args.device
    )
    model_name = "microsoft/resnet-18"
    cache = Path.home() / ".cache" / "hw0_resnet"
    print(f"Device: {device}\nModel: {model_name}", flush=True)
    processor = AutoImageProcessor.from_pretrained(
        model_name, cache_dir=str(cache / "models")
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_name, cache_dir=str(cache / "models")
    ).to(device)
    model.eval()
    dataset = MNIST(root=str(cache / "datasets"), train=False, download=True)
    total = min(args.limit, len(dataset)) if args.limit else len(dataset)
    print(f"Test samples: {total}", flush=True)
    print("Metric: raw ImageNet class-ID match against MNIST digit labels.\n"
          "WARNING: different label semantics; not digit-recognition accuracy.",
          flush=True)

    correct = 0
    started = time.perf_counter()
    with torch.inference_mode(), tqdm(total=total, unit="image") as progress:
        for begin in range(0, total, args.batch_size):
            samples = [dataset[i] for i in range(
                begin, min(begin + args.batch_size, total)
            )]
            images = [image.convert("RGB") for image, _ in samples]
            labels = torch.tensor([label for _, label in samples])
            # The model's processor performs resize, crop and normalization.
            inputs = processor(images=images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            if begin == 0:
                tqdm.write(f"Input tensor shape: {tuple(pixels.shape)}")
            predictions = model(pixel_values=pixels).logits.argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()
            progress.update(len(samples))

    accuracy = 100.0 * correct / total
    result = {
        "model": model_name,
        "device": str(device),
        "dataset": "MNIST test",
        "samples": total,
        "metric": "raw ImageNet class-ID match against MNIST digit labels",
        "label_spaces_aligned": False,
        "correct": correct,
        "accuracy_percent": accuracy,
        "seconds": round(time.perf_counter() - started, 3),
        "torch_version": torch.__version__,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print('\nSuggested commit message:\n'
          f'Run pretrained ResNet-18 on MNIST; raw-index accuracy '
          f'{accuracy:.4f}% on {total} test images')


if __name__ == "__main__":
    main()
