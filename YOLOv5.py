from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

model = YOLO("yolov5m.pt")  # Medium version

def load_images_from_folder(folder, max_images=100):
    images = []
    for i, filename in enumerate(sorted(os.listdir(folder))):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')) and i < max_images:
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            images.append((img, filename))
    return images

images = load_images_from_folder("images")

for img, filename in images:
    results = model(img)[0]
    result_img = results.plot()
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(filename)
    plt.axis("off")
    plt.show()
