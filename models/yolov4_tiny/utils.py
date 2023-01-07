import cv2
import numpy as np


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
