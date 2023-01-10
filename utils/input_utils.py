from pathlib import Path
from typing import Union
import numpy as np
import cv2


class ImageReader:
    """Iterate over all images (.jpg or .jpeg) in folder and return original image
        NOTE: The ImageReader may work fine with other formats of images but it's not a guarantee
    """
    def __init__(self, folder_path: Union[Path, str]) -> None:
        """
        Parameter
        ---------
        folder_path : Union[Path, str]
            Path to folder with images
        """
        folder_path = Path(folder_path)
        assert folder_path.is_dir(), f'No such dir: {folder_path}'
        
        self.folder_path = folder_path
        self.img_path_gen = self.folder_path.glob('*')
        self.images_count = len(list(self.folder_path.glob('*')))
    
    def __len__(self):
        return self.images_count

    def __iter__(self):
        return self
    
    def __next__(self) -> np.ndarray:
        """        
        Returns
        -------
        np.ndarray
            Raw image in BGR format with shape of (height, width, channels)
        """
        img_path = next(self.img_path_gen)
        image = cv2.imread(filename=img_path.as_posix())
        return image

class VideoReader:
    """Iterate over video (.mp4) and return original frames
        NOTE: The VideoReader may work fine with other formats of video but it's not a guarantee
    """
    def __init__(self, 
                 video_path: Union[Path, str]) -> None:
        """
        Parameter
        ---------
        video_path : Union[Path, str]
            Path to video
        """
        video_path = Path(video_path)
        assert video_path.is_file(), f'No such file: {video_path}'

        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path.as_posix())
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self):
        return self.total_frames

    def __iter__(self):
        return self
    
    def __next__(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Frame in BGR format from video stream with shape of (height, width, channels)
        """
        cap_success, frame = self.video_capture.read()
        if cap_success:
            return frame
        self.video_capture.release()
        raise StopIteration()        
