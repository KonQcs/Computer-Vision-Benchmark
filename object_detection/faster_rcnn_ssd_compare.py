import argparse
from pathlib import Path
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn,
    ssd300_vgg16,
)


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_model_pair(device):
    faster_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ssd_weights = SSD300_VGG16_Weights.DEFAULT

    faster_model = fasterrcnn_resnet50_fpn(weights=faster_weights).to(device).eval()
    ssd_model = ssd300_vgg16(weights=ssd_weights).to(device).eval()

    return (
        faster_model,
        faster_weights.meta["categories"],
        faster_weights.transforms(),
        ssd_model,
        ssd_weights.meta["categories"],
        ssd_weights.transforms(),
    )


def detect(model, image, preprocess, categories, device, threshold):
    image_tensor = preprocess(image).to(device)
    with torch.no_grad():
        start = time.time()
        prediction = model([image_tensor])[0]
        duration = time.time() - start

    results = []
    for box, label, score in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"]
    ):
        if score >= threshold:
            label_idx = int(label.item())
            label_name = (
                categories[label_idx]
                if label_idx < len(categories)
                else f"unknown_{label_idx}"
            )
            results.append((box.tolist(), label_name, float(score.item())))
    return results, duration


def draw_boxes(ax, image, detections, title, duration):
    ax.imshow(image)
    for box, label, score in detections:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1,
            f"{label} {score:.2f}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "green", "alpha": 0.5},
        )
    ax.set_title(f"{title}\nTime: {duration:.3f} sec")
    ax.axis("off")


def image_files(folder: Path):
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Faster R-CNN and SSD300.")
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
    (
        faster_model,
        faster_categories,
        faster_preprocess,
        ssd_model,
        ssd_categories,
        ssd_preprocess,
    ) = load_model_pair(device)

    files = image_files(args.images)
    if not files:
        raise RuntimeError(f"No images found in {args.images}")

    for image_path in files:
        image = Image.open(image_path).convert("RGB")
        faster_results, faster_time = detect(
            faster_model,
            image,
            faster_preprocess,
            faster_categories,
            device,
            args.threshold,
        )
        ssd_results, ssd_time = detect(
            ssd_model,
            image,
            ssd_preprocess,
            ssd_categories,
            device,
            args.threshold,
        )

        print(
            f"{image_path.name}: "
            f"Faster R-CNN={len(faster_results)} detections, "
            f"SSD300={len(ssd_results)} detections"
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        draw_boxes(axes[0], image, faster_results, "Faster R-CNN", faster_time)
        draw_boxes(axes[1], image, ssd_results, "SSD300", ssd_time)
        plt.suptitle(f"Detection results: {image_path.name}", fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

        if args.save_dir:
            output_path = args.save_dir / f"{image_path.stem}_comparison.png"
            fig.savefig(output_path, dpi=160)

        if args.no_show:
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
