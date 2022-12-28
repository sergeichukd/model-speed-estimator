import cv2
from pathlib import Path
import numpy as np


# TODO Change name
def read_and_preprocess_image(image_path: Path, out_size=(416, 416)) -> np.ndarray:
    # Read image in BGR format, because person-detection-0200 takes input in BGR format
    image = cv2.imread(filename=image_path.as_posix())
    image = cv2.resize(image, dsize=out_size)

    #  Transpose and add new dimension to correcpond model input format (B, C, H, W)
    return image
