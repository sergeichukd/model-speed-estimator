from pathlib import Path
from typing import Union, Callable
import numpy as np
import cv2


# TODO: check that path is valid
# TODO: describe, what image formats are supported
class ImageReader:
    """Iterate over all images in folder and return preprocessed images as input of a neural network.
    """
    def __init__(self, 
                 folder_path: Union[Path, str], 
                 preprocess_image_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Parameter
        ---------
        folder_path : Union[Path, str]
            Path to folder with images.

        preprocess_image_fn : Callable[[np.ndarray], np.ndarray]
            Image preprocessing function before feeding it to a neural network.
            
            Parameter
            ---------
            image : np.ndarray
                Image in BGR format with shape of (height, width, channels).
                
            Returns
            -------
            np.ndarray
                Image in valid format as input of the neural network
        """
        self.folder_path = Path(folder_path)
        self.img_path_gen = self.folder_path.glob('*')
        self.images_count = len(list(self.folder_path.glob('*')))
        self.preprocess_image_fn = preprocess_image_fn
    
    def __len__(self):
        return self.images_count

    def __iter__(self):
        return self
    
    def __next__(self):
        img_path = next(self.img_path_gen)
        image = cv2.imread(filename=img_path.as_posix())
        return self.preprocess_image_fn(image)
