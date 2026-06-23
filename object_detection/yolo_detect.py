import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def default_model_path() -> str:
    local_model = PROJECT_DIR / "models" / "yolov5mu.pt"
    return str(local_model) if local_model.exists() else "yolov5mu.pt"


def load_images_from_folder(folder: Path, max_images: int):
    images = []
    for filename in sorted(folder.iterdir()):
        if filename.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(filename))
        if image is not None:
            images.append((image, filename.name))
        if len(images) >= max_images:
            break
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv5 detection on the image folder.")
    parser.add_argument("--images", type=Path, default=PROJECT_DIR / "images")
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true", help="Skip matplotlib windows.")
    args = parser.parse_args()

    if not args.images.exists():
        raise FileNotFoundError(f"Image folder not found: {args.images}")
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    images = load_images_from_folder(args.images, args.max_images)
    if not images:
        raise RuntimeError(f"No images found in {args.images}")

    for image, filename in images:
        result = model(image)[0]
        result_image = result.plot()
        print(f"{filename}: {len(result.boxes)} detections")

        if args.save_dir:
            cv2.imwrite(str(args.save_dir / filename), result_image)

        if not args.no_show:
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.title(filename)
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
