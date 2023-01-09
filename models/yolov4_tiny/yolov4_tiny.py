import cv2
from pathlib import Path


MODEL_DIR = Path('vendor/models/yolov4-tiny')
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# TODO: Remove or rename accuracy='FP32', device: str='CPU'
# TODO: Check input size
class YoloV4Tiny:
    # MAKE proper name for "accuracy" and "device" variable
    def __init__(self, accuracy='FP32', device: str='CPU', input_size = (416, 416), name='yolov4_tiny') -> None:
        weights_path = MODEL_DIR / 'yolov4-tiny.weights'
        config_path = MODEL_DIR / 'yolov4-tiny.cfg'
        net = cv2.dnn.readNet(weights_path.as_posix(), config_path.as_posix())
        
        # TODO: Разобраться с setPreferableBackend и setPreferableTarget (FP32 or FP16)
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CPU)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=input_size, scale=1/255, swapRB=True) # TODO Разобраться со swapRB: это BGR или RGB в итоге?
        self.name = name

    def infer(self, image):
        """
        Return: classes, scores, boxes
        """
        classes, scores, boxes = self.model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        return classes, scores, boxes
