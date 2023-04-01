import base64
import cv2
from ultralytics import YOLO
import numpy as np

# Load a model
# load a pretrained model (recommended for training)
model = YOLO("widerface.pt")
# res = model("./images/test.jpg")

def detect(img):
    img = np.array(img)
    # Use the model
    res = model(img)  # predict on an image
    plot = res[0].plot()
    boxes = res[0].boxes.cpu().numpy().boxes.tolist()

    return boxes, plot
