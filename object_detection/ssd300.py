import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from PIL import Image
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def draw_predictions(ax, image, prediction, categories, threshold):
    ax.imshow(image)
    for box, label, score in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"]
    ):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        label_idx = int(label.item())
        label_name = (
            categories[label_idx] if label_idx < len(categories) else f"unknown_{label_idx}"
        )
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1,
            f"{label_name} {float(score):.2f}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "green", "alpha": 0.5},
        )
    ax.axis("off")


def image_files(folder: Path):
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SSD300 object detection.")
    parser.add_argument("--images", type=Path, default=PROJECT_DIR / "images")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true", help="Skip matplotlib windows.")
    args = parser.parse_args()

    if not args.images.exists():
        raise FileNotFoundError(f"Image folder not found: {args.images}")
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    files = image_files(args.images)
    if not files:
        raise RuntimeError(f"No images found in {args.images}")

    for image_path in files:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).to(device)
        with torch.no_grad():
            prediction = model([image_tensor])[0]

        detections = int((prediction["scores"] >= args.threshold).sum().item())
        print(f"{image_path.name}: {detections} detections")

        fig, ax = plt.subplots(figsize=(10, 7))
        draw_predictions(ax, image, prediction, categories, args.threshold)
        ax.set_title(image_path.name)

        if args.save_dir:
            fig.savefig(args.save_dir / f"{image_path.stem}_ssd300.png", dpi=160)

        if args.no_show:
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
