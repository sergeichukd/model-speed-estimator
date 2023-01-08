from typing import List, Tuple
from entities.detection import Detect, BBox
import numpy as np
import cv2
from random import randint


def rel_to_abs_coordinates(x_min: float, y_min: float, x_max: float, y_max: float, img_size: Tuple[int, int]):
    """
    img_size - Image size (height, width)
    """
    img_height, img_width = img_size

    x_min_absolute = int(x_min * img_width)
    x_max_absolute = int(x_max * img_width)
    y_min_absolute = int(y_min * img_height)
    y_max_absolute = int(y_max * img_height)
    return (x_min_absolute, y_min_absolute), (x_max_absolute, y_max_absolute)

def abs_to_rel_coordinates(x_min: float, y_min: float, x_max: float, y_max: float, img_size: Tuple[int, int]):
    """
    img_size - Image size (height, width)
    """
    img_height, img_width = img_size

    x_min_relative = x_min / img_width
    x_max_relative = x_max / img_width
    y_min_relative = y_min / img_height
    y_max_relative = y_max / img_height
    return (x_min_relative, y_min_relative), (x_max_relative, y_max_relative)

def tlwh_to_tlbr(x_min, y_min, w, h):
    return (x_min, y_min), (x_min + w, y_min + h)

# TODO: What color format does get this function? BGR or RGB
def visualize_detections(img: np.ndarray, detections: List[Detect]) -> np.ndarray:
    """
    img - image in BGR format
    """
    output_imgage_size = (720, 720)
    new_img = img.copy()
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    for detection in detections:
        top_left, bottom_right = rel_to_abs_coordinates(detection.bbox.x_min, 
                                                        detection.bbox.y_min, 
                                                        detection.bbox.x_max, 
                                                        detection.bbox.y_max, 
                                                        img_size = output_imgage_size)
        color = (0,0,255) # Red in BGR
        text = f'{detection.class_name}: {detection.conf:.3f}'
        new_img = cv2.resize(new_img, output_imgage_size)
        new_img = cv2.rectangle(new_img, 
                                pt1=(top_left[0], top_left[1] + 5), 
                                pt2=bottom_right, 
                                color=color, 
                                thickness=2)
        new_img = cv2.putText(new_img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return new_img

def save_results():
    pass