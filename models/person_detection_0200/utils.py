import cv2
import numpy as np


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
            * Color format: BGR (person-detection-0200 takes input in BGR format)
            * Image size: 256 x 256
            * Output shape: (batch_size = 1, channels, height, width)
    """

    image = cv2.resize(image, dsize=(256, 256))
    image = image.transpose(2, 0, 1)[np.newaxis, ...]
    return image
