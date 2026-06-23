import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision import transforms
import cv2
import os
import matplotlib.pyplot as plt

# Λίστα κατηγοριών από το COCO
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#default βάρη
weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights).to(device).eval()

#τιμές (mean/std)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#load"images"
def load_images_from_folder(folder, max_images=100):
    images = []
    for i, filename in enumerate(sorted(os.listdir(folder))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and i < max_images:
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append((img, filename))
    return images

#boxes
def draw_predictions(img, boxes, labels, scores, threshold=0.6):
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold and 0 < label < len(COCO_CLASSES):
            x1, y1, x2, y2 = map(int, box)
            label_name = COCO_CLASSES[label]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label_name} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

#↓↓ main ↓↓
#output
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


images = load_images_from_folder("images")

for img, filename in images:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)[0]

    #plot
    result_img = draw_predictions(img.copy(), output["boxes"], output["labels"], output["scores"])

    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(filename)
    plt.axis("off")
    plt.show()

    #store
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, result_img)
