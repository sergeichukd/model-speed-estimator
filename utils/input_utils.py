from pathlib import Path
from typing import Union, Callable, Tuple
import numpy as np
import cv2


# TODO: check that path is valid
# TODO: describe, what image formats are supported
class ImageReader:
    """Iterate over all images in folder and return original and preprocessed images as input of a neural network.
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
        folder_path = Path(folder_path)
        assert folder_path.is_dir(), f'No such dir: {folder_path}'
        
        self.folder_path = folder_path
        self.img_path_gen = self.folder_path.glob('*')
        self.images_count = len(list(self.folder_path.glob('*')))
        self.preprocess_image_fn = preprocess_image_fn
    
    def __len__(self):
        return self.images_count

    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns Tuple (image_path, original_image, preprocessed_image)
        """
        img_path = next(self.img_path_gen)
        image = cv2.imread(filename=img_path.as_posix())
        return image, self.preprocess_image_fn(image)

# TODO: describe, what video formats are supported
# TODO Check wheather BGR format video capture return or not
class VideoReader:
    """Iterate over video and return original and preprocessed frames as input of a neural network.
    """
    def __init__(self, 
                 video_path: Union[Path, str], 
                 preprocess_frame_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Parameter
        ---------
        video_path : Union[Path, str]
            Path to video.

        preprocess_frame_fn : Callable[[np.ndarray], np.ndarray]
            Frame preprocessing function before feeding it to a neural network.
            
            Parameter
            ---------
            image : np.ndarray
                Image in BGR format with shape of (height, width, channels).
                
            Returns
            -------
            np.ndarray
                Image in valid format as input of the neural network
        """
        video_path = Path(video_path)
        assert video_path.is_file(), f'No such file: {video_path}'

        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path.as_posix())
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        self.preprocess_frame_fn = preprocess_frame_fn
        self.frame_counter = -1

    def __len__(self):
        return self.total_frames

    def __iter__(self):
        return self
    
    # Make right docstring
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns Tuple (frame_number, original_frame, preprocessed_frame)
        """
        cap_success, frame = self.video_capture.read()
        self.frame_counter += 1
        if cap_success:
            return frame, self.preprocess_frame_fn(frame)
        self.video_capture.release()
        raise StopIteration()        
