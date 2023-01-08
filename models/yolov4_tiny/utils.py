import cv2
import numpy as np
from typing import Tuple
from entities.detection import Detect, BBox
from utils.output_utils import tlwh_to_tlbr, abs_to_rel_coordinates


# TODO Check yolov4-tiny input format
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Image preprocessing before feeding it to a neural network.

    Parameter
    ---------
    image : np.ndarray
        Image in BGR format with shape of (height, width, channels).

    Returns
    -------
    np.ndarray
        Image for neural network in valid format:
            * Color format: BGR 
            * Image size: 416 x 416
            * Output shape: (height, width, channels)
    """

    image = cv2.resize(image, dsize=(416, 416))
    return image

def unify_prediction(preds: Tuple[np.ndarray, np.ndarray, np.ndarray], img_size: Tuple[int, int]) -> Detect:
    """
    Convert model prediction to list of detections

    Parameter
    ---------
    preds : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Prediction of model for one image as tuple of np.ndarrays: (labels, scores, bboxes), where:
            * labels : np.ndarray with shape (num_of_bboxes) - Labels of predicted bboxes
            * scores : np.ndarray with shape (num_of_bboxes) - Scores of predicted bboxes
            * bboxes : np.ndarray with shape (num_of_bboxes, 4) - Bboxes with coordinates (x_min, y_min, width, height)
    
    img_size : Tuple[int, int]
        The size of the image in format (height, width) for which the bbox predictions are made.

    Returns
    -------
    List[Detect]
        List of detections in unified format
    """
    class_names = []
    with open("models/yolov4_tiny/coco-classes.txt", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    
    detections = []
    for label, score, bbox in zip(*preds):
        (x_min, y_min), (x_max, y_max) = tlwh_to_tlbr(*bbox)
        (x_min, y_min), (x_max, y_max) = abs_to_rel_coordinates(x_min, y_min, x_max, y_max, img_size)
        detections.append(Detect(
            bbox=BBox(x_min, y_min, x_max, y_max),
            conf=score,
            label=label,
            class_name=class_names[label]
        ))
    return detections
