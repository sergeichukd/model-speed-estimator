import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
from pathlib import Path
from tqdm import tqdm

from utils.input_utils import ImageReader

from models.person_detection_0200.person_detection_0200 import PersonDetection0200
from models.person_detection_0200.utils import preprocess_image as preprocess_image_person_detection_0200

from models.yolov4_tiny import YoloV4Tiny
from models.yolov4_tiny.utils import preprocess_image as preprocess_image_yolov4_tiny


image_dir_path = Path('datasets/images')

image_reader_person_detection_0200 = ImageReader(image_dir_path, preprocess_image_person_detection_0200)
person_detection_0200 = PersonDetection0200()

for img in tqdm(image_reader_person_detection_0200):
    preds = person_detection_0200.infer(img)
    # print(preds.shape)  


yolov4_tiny = YoloV4Tiny()

image_reader_yolov4_tiny = ImageReader(image_dir_path, preprocess_image_yolov4_tiny)

for img in tqdm(image_reader_yolov4_tiny):
    classes, scores, boxes = yolov4_tiny.infer(img)

    # print('classes:\n', classes)
    # print('scores:\n', scores)
    # print('boxes:\n', boxes)
