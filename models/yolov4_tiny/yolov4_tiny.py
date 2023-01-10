import cv2
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple
from entities.detection import Detect, BBox
from utils.output_utils import tlwh_to_tlbr, abs_to_rel_coordinates
from models.abstract_model import Model


MODEL_DIR = Path('vendor/models/yolov4-tiny')

class YoloV4Tiny(Model):
    def __init__(self, 
                 input_size = (416, 416), 
                 name='yolov4_tiny', 
                 confidence_threshold=0.5, 
                 nms_threshold=0.5) -> None:
        super().__init__()

        weights_path = MODEL_DIR / 'yolov4-tiny.weights'
        config_path = MODEL_DIR / 'yolov4-tiny.cfg'
        net = cv2.dnn.readNet(weights_path.as_posix(), config_path.as_posix())
        
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=input_size, scale=1/255, swapRB=True)
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def infer(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make prediction
        Parameter
        ---------
        image : np.ndarray
            Imput image for neural network in valid format:
                * Color format: BGR 
                * Image size: 416 x 416
                * Shape: (height, width, channels)
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Prediction of model for one image as tuple of np.ndarrays: (labels, scores, bboxes), where:
                * labels : np.ndarray with shape (num_of_bboxes) - Labels of predicted bboxes
                * scores : np.ndarray with shape (num_of_bboxes) - Scores of predicted bboxes
                * bboxes : np.ndarray with shape (num_of_bboxes, 4) - Bboxes with coordinates (x_min, y_min, width, height)
        """
        classes, scores, boxes = self.model.detect(image, self.confidence_threshold, self.nms_threshold)
        return classes, scores, boxes

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
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

    def unify_prediction(self, preds: Tuple[np.ndarray, np.ndarray, np.ndarray], img_size: Tuple[int, int]) -> Detect:
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
