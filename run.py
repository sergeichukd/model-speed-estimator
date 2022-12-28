import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
from pathlib import Path

from models.person_detection_0200.person_detection_0200 import PersonDetection0200
from models.person_detection_0200.utils import read_and_preprocess_image as read_and_preprocess_image_person_detection

from models.yolov4_tiny import YoloV4Tiny
from models.yolov4_tiny.utils import read_and_preprocess_image as read_and_preprocess_image_yolov4


person_detection_0200 = PersonDetection0200()
image_path = Path('datasets/people.jpg')
input_image = read_and_preprocess_image_person_detection(image_path)
preds = person_detection_0200.infer(input_image)

print(preds.shape)



yolo = YoloV4Tiny()
input_image = read_and_preprocess_image_yolov4(image_path)
classes, scores, boxes = yolo.infer(input_image)
print('classes:\n', classes)
print('scores:\n', scores)
print('boxes:\n', boxes)
