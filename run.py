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

import models
from utils.output_utils import visualize_detections

from matplotlib import pyplot as plt
import time
import os

from typing import Union


PROCESS_VIDEO = True
SAVE_VISUALIZATION = True

DATA_PATH = Path('datasets/videos/people_20s.mp4')
# DATA_PATH = Path('datasets/images')



Reader: Union[VideoReader, ImageReader]

if PROCESS_VIDEO:
    Reader = VideoReader
    # RESULTS_PATH = Path(f'results/videos_{time.strftime("%d-%m-%Y_%H-%M-%S")}')
    RESULTS_PATH = Path(f'results/videos')
else:
    Reader = ImageReader
    # RESULTS_PATH = Path(f'results/images_{time.strftime("%d-%m-%Y_%H-%M-%S")}')
    RESULTS_PATH = Path(f'results/images')

# os.mkdir(RESULTS_PATH)

image_reader_person_detection_0200 = Reader(DATA_PATH, preprocess_image_person_detection_0200)
person_detection_0200 = models.person_detection_0200.PersonDetection0200()
model_results_path = RESULTS_PATH / 'person_detection_0200'
# os.mkdir(model_results_path)

if PROCESS_VIDEO:
    video_path = model_results_path / DATA_PATH.with_suffix('.avi').name
    print(type(video_path.as_posix()))
    print(video_path.as_posix())
    # os.makedirs('results/videos_08-01-2023_21-17-50/person_detection_0200')
    out_video = cv2.VideoWriter(
                                # filename='results/videos_08-01-2023_21-17-50/person_detection_0200/people_20s.avi', 
                                # filename='test.avi', 
                                filename=video_path.as_posix(), 
                                fourcc=cv2.VideoWriter_fourcc(*'FMP4'), 
                                fps=25, 
                                # fps=image_reader_person_detection_0200.fps, 
                                frameSize=(720, 720),
                                # frameSize=(image_reader_person_detection_0200.frame_width, image_reader_person_detection_0200.frame_height),
                                isColor=True)

# TODO: Rename img_path to correspond for frame id and image path
for img_path, orig_img, img in tqdm(image_reader_person_detection_0200):
    preds = person_detection_0200.infer(img)
    dets = models.person_detection_0200.utils.unify_prediction(preds, img_size=orig_img.shape[:2])
    vis_img = visualize_detections(orig_img, dets)
    if PROCESS_VIDEO:
        print(vis_img.shape)
        # plt.imshow(vis_img)
        # plt.show()
        out_video.write(vis_img)
    else:
        image_path = model_results_path / Path(img_path).name
        print(f'Write to: {image_path}')
        cv2.imwrite(image_path.as_posix(), vis_img)
    # plt.imshow(vis_img)
    # plt.show()
if PROCESS_VIDEO:
    out_video.release()


# yolov4_tiny_model = YoloV4Tiny()

# image_reader_yolov4_tiny = Reader(DATA_PATH, preprocess_image_yolov4_tiny)

# model_results_path = RESULTS_PATH / 'yolov4_tiny'
# os.makedirs(model_results_path, exist_ok=False)

# for img_path, orig_img, img in tqdm(image_reader_yolov4_tiny):
#     preds = yolov4_tiny_model.infer(img)
#     dets = models.yolov4_tiny.utils.unify_prediction(preds, img.shape[:2])
#     vis_img = visualize_detections(orig_img, dets)
#     if PROCESS_VIDEO:
#         pass
#     else:
#         image_path = model_results_path / Path(img_path).name
#         print(f'Write to: {image_path}')
#         cv2.imwrite(image_path.as_posix(), vis_img)
#     # plt.imshow(vis_img)
#     # plt.show()
#     # cv2.waitKey(1)
#     # cv2.destroyAllWindows()
