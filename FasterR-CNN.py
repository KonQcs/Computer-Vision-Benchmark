import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
import cv2
import os
import matplotlib.pyplot as plt

# COCO labels
COCO_CLASSES = [ "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()

# Transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load images from folder
def load_images_from_folder(folder, max_images=100):
    images = []
    for i, filename in enumerate(sorted(os.listdir(folder))):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')) and i < max_images:
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append((img, filename))
    return images

# Draw predictions on image
def draw_boxes(img, boxes, labels, scores, threshold=0.5):
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box)
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

# Run detection
images = load_images_from_folder("images")

for img, filename in images:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)[0]

    result_img = draw_boxes(img.copy(), output["boxes"], output["labels"], output["scores"])

    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(filename)
    plt.axis("off")
    plt.show()
