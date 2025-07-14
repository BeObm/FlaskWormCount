import yaml
import cv2
import numpy as np

def setup_dataset():
    data_config = {
        "path": "./worm_dataset",
        "train": "train/images",
        "val": "val/images",
        "test": "val/images",
        "names": ["worm"],
        "nc": 1
    }
    with open("dataset.yaml", "w") as f:
        yaml.dump(data_config, f)
    print("✅ dataset.yaml created")

def draw_detections(result):
    img = result.orig_img.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = result.names[int(box.cls[0])]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img
