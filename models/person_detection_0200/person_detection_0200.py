from openvino.runtime import Core
from pathlib import Path
import cv2
import numpy as np
from entities.detection import Detect, BBox
from typing import List, Tuple
from models.abstract_model import Model


MODEL_DIR = Path('vendor/models/intel/person-detection-0200')

class PersonDetection0200(Model):
    def __init__(self, 
                 accuracy: str='FP32', 
                 device: str='CPU', 
                 name: str ='person_detection_0200', 
                 confidence_threshold: float = 0.5) -> None:
        super().__init__()

        ie = Core()
        model_path = MODEL_DIR / accuracy / 'person-detection-0200.xml'
        self.model = ie.read_model(model=model_path.as_posix())
        self.compiled_model = ie.compile_model(
            model=self.model, 
            device_name=device.upper(),
            config={'PERFORMANCE_HINT': 'THROUGHPUT'},
        )
        self.output_layer = self.compiled_model.output(0)
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.model_dir = MODEL_DIR

    
    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Parameter
        ---------
        image : np.ndarray
            Input image for neural network in valid format:
                * Color format: BGR
                * Image size: 256 x 256
                * Shape: (batch_size = 1, channels, height, width)

        Returns
        -------
        np.ndarray
        Prediction of model for one image with shape: (1, 1, 200, 7) in the format (1, 1, N, 7),
        where N is the number of detected bounding boxes. 
        Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max], where:
            * image_id - ID of the image in the batch
            * label - predicted class ID (0 - person)
            * conf - confidence for the predicted class
            * (x_min, y_min) - coordinates of the top left bounding box corner
            * (x_max, y_max) - coordinates of the bottom right bounding box corner
        """
        return self.compiled_model([image])[self.output_layer]

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
                * Color format: BGR (person-detection-0200 takes input in BGR format)
                * Image size: 256 x 256
                * Output shape: (batch_size = 1, channels, height, width)
        """

        image = cv2.resize(image, dsize=(256, 256))
        image = image.transpose(2, 0, 1)[np.newaxis, ...]
        return image


    def unify_prediction(self, predictions: np.ndarray, img_size: Tuple[int, int]) -> List[Detect]:
        """
        Convert model prediction to list of detections

        Parameter
        ---------
        predictions : np.ndarray
            Prediction of model for one image with shape: (1, 1, 200, 7) in the format (1, 1, N, 7),
            where N is the number of detected bounding boxes. 
            Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max], where:
                * image_id - ID of the image in the batch
                * label - predicted class ID (0 - person)
                * conf - confidence for the predicted class
                * (x_min, y_min) - coordinates of the top left bounding box corner
                * (x_max, y_max) - coordinates of the bottom right bounding box corner
        
        img_size : Tuple[int, int]
            Image size: (height, width)

        Returns
        -------
        List[Detect]
            List of detections in unified format
        """
        detections = []
        for detect in predictions.squeeze():
            image_id, label, conf, x_min, y_min, x_max, y_max = detect

            if conf > self.confidence_threshold:
                detections.append(Detect(
                    bbox=BBox(x_min, y_min, x_max, y_max),
                    conf=conf,
                    label=label,
                    class_name='Person'
                ))
        return detections
