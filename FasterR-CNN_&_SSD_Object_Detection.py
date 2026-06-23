import torch
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, SSD300_VGG16_Weights, fasterrcnn_resnet50_fpn, ssd300_vgg16
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time

# labels Faster R-CNN
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# labels SSD300
VOC_LABELS = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


transform = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models
faster_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)


faster_model.to(device)
faster_model.eval()

ssd_model.to(device)
ssd_model.eval()


def detect(model, image_tensor, threshold=0.5, label_set="coco"):
    with torch.no_grad():
        start = time.time()
        prediction = model(image_tensor.to(device))[0]
        duration = time.time() - start

    boxes, labels, scores = prediction["boxes"], prediction["labels"], prediction["scores"]

    results = []
    for i in range(len(scores)):
        if scores[i] >= threshold:
            box = boxes[i].tolist()
            label_idx = labels[i].item()
            if label_set == "coco":
                label = COCO_INSTANCE_CATEGORY_NAMES[label_idx] if label_idx < len(COCO_INSTANCE_CATEGORY_NAMES) else f"unknown_{label_idx}"
            else:
                label = VOC_LABELS[label_idx] if label_idx < len(VOC_LABELS) else f"unknown_{label_idx}"
            score = scores[i].item()
            results.append((box, label, score))
    return results, duration

# VS plots
def draw_boxes(ax, image, detections, title, duration):
    ax.imshow(image)
    for box, label, score in detections:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f"{label} {score:.2f}", color="white",
                fontsize=8, bbox=dict(facecolor="green", alpha=0.5))
    ax.set_title(f"{title}\nTime: {duration:.3f} sec")
    ax.axis("off")


#↓↓main↓↓
#load images
image_dir = "images"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]


for file in image_files:
    image_path = os.path.join(image_dir, file)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)


    detections_faster, time_faster = detect(faster_model, image_tensor, label_set="coco")
    detections_ssd, time_ssd = detect(ssd_model, image_tensor, label_set="voc")


    image_faster = image.copy()
    image_ssd = image.copy()

    #VS plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw_boxes(axes[0], image_faster, detections_faster, "Faster R-CNN", time_faster)
    draw_boxes(axes[1], image_ssd, detections_ssd, "SSD300 (VOC)", time_ssd)

    plt.suptitle(f"Ανίχνευση για: {file}", fontsize=14)
    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    plt.show()
