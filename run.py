import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
from pathlib import Path
from tqdm import tqdm

from utils.input_utils import ImageReader, VideoReader

from models.person_detection_0200.person_detection_0200 import PersonDetection0200
# from models.person_detection_0200.utils import preprocess_image as preprocess_image_person_detection_0200

from models.yolov4_tiny import YoloV4Tiny
# from models.yolov4_tiny.utils import preprocess_image as preprocess_image_yolov4_tiny

import models
from utils.output_utils import visualize_detections, ImageWriter, VideoWriter, OUTPUT_IMAGE_SIZE

from matplotlib import pyplot as plt
import time
import os

from typing import Union
import distutils.dir_util


PROCESS_VIDEO = False
SAVE_VISUALIZATION = True


if PROCESS_VIDEO:
    DATA_PATH = Path('datasets/videos/people_20s.mp4')
else:
    DATA_PATH = Path('datasets/images')

Reader: Union[VideoReader, ImageReader]

if PROCESS_VIDEO:
    Reader = VideoReader
    RESULTS_PATH = Path(f'results/videos_{time.strftime("%d-%m-%Y_%H-%M-%S")}')
else:
    Reader = ImageReader
    RESULTS_PATH = Path(f'results/images_{time.strftime("%d-%m-%Y_%H-%M-%S")}')


def estimate_model(model, data_reader: Reader):
    model_results_path = RESULTS_PATH / model.name
    distutils.dir_util.mkpath(model_results_path.as_posix())

    if PROCESS_VIDEO:
        video_path = model_results_path / DATA_PATH.with_suffix('.avi').name
        writer = VideoWriter(video_path, 
                            fps=data_reader.fps, 
                            frame_size=OUTPUT_IMAGE_SIZE)
    else:
        writer = ImageWriter(model_results_path)

    for orig_img in tqdm(data_reader):
        input_img = model.preprocess_image(orig_img)
        preds = model.infer(input_img)
        dets = model.unify_prediction(preds, input_img.shape[:2])
        vis_img = visualize_detections(orig_img, dets)
        writer.write(vis_img)
    writer.close()


# #  Estimate person_detection_0200
data_reader = Reader(DATA_PATH)
model = PersonDetection0200()
estimate_model(model, data_reader)

# # Estimate yolov4_tiny
data_reader = Reader(DATA_PATH)
model = YoloV4Tiny()
estimate_model(model, data_reader)
