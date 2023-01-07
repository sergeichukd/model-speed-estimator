import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
from pathlib import Path
from tqdm import tqdm

from utils.input_utils import ImageReader, VideoReader

from models.person_detection_0200.person_detection_0200 import PersonDetection0200
from models.person_detection_0200.utils import preprocess_image as preprocess_image_person_detection_0200

from models.yolov4_tiny import YoloV4Tiny
from models.yolov4_tiny.utils import preprocess_image as preprocess_image_yolov4_tiny

PROCESS_VIDEO = True
DATA_PATH = Path('datasets/videos/people_05s.mp4')

if PROCESS_VIDEO:
    Reader = VideoReader
else:
    Reader = ImageReader

image_reader_person_detection_0200 = Reader(DATA_PATH, preprocess_image_person_detection_0200)
person_detection_0200 = PersonDetection0200()

for img in tqdm(image_reader_person_detection_0200):
    preds = person_detection_0200.infer(img)
    # print(preds.shape)  


yolov4_tiny = YoloV4Tiny()

image_reader_yolov4_tiny = Reader(DATA_PATH, preprocess_image_yolov4_tiny)

for img in tqdm(image_reader_yolov4_tiny):
    classes, scores, boxes = yolov4_tiny.infer(img)

    # print('classes:\n', classes)
    # print('scores:\n', scores)
    # print('boxes:\n', boxes)
