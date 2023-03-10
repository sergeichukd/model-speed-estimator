from typing import List, Tuple, Union
from entities.detection import Detect
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from tabulate import tabulate


OUTPUT_IMAGE_SIZE = (720, 720)

class VideoWriter:
    def __init__(self, video_path: Union[Path, str], fps: float, frame_size: Tuple[int, int]) -> None: 
        self.video_path = Path(video_path)
        self.fps = fps
        self.frame_size = frame_size
        self.out_video = cv2.VideoWriter(filename=str(video_path), 
                                         fourcc=cv2.VideoWriter_fourcc(*'FMP4'), 
                                         fps=fps, 
                                         frameSize=frame_size,
                                         isColor=True)
        print('Write video to:', self.video_path)

    def write(self, img):
        self.out_video.write(img)
    
    def close(self):
        self.out_video.release()
    
class ImageWriter:
    def __init__(self, save_path: Union[Path, str]):
        self.save_path = Path(save_path)
        self.image_counter = 0
        print('Write images to:', self.save_path)
    
    def write(self, img: np.ndarray) -> None:
        img_path = self.save_path / f'{self.image_counter}.jpg'
        cv2.imwrite(img_path.as_posix(), img)
        self.image_counter += 1
    
    def close(self):
        pass

@dataclass
class Timings:
    preprocess_time = 0
    inference_time = 0
    postprocess_time = 0

    def total_time(self):
        return self.preprocess_time + self.inference_time + self.postprocess_time

    def percents(self) -> Tuple[float, float, float]:
        tot_time = self.total_time()
        return (
            self.preprocess_time / tot_time, 
            self.inference_time / tot_time, 
            self.postprocess_time / tot_time)

def rel_to_abs_coordinates(x_min: float, y_min: float, 
                           x_max: float, y_max: float, 
                           img_size: Tuple[int, int]
                           ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Convert relative to absolute rectangle coordinates        
    Parameter
    -------
    x_min : float, y_min : float, x_max : float, y_max : float
        Top-left and right-bottom relative coordinates of rectangle
     
    img_size : Tuple[int, int]
        Image size (height, width)
    
    Returns
    -------
    Tuple[Tuple[int, int], Tuple[int, int]]
        Rectangle absolute coordinates in format ((x_min, y_min), (x_max, y_max))
    """
    img_height, img_width = img_size

    x_min_absolute = int(x_min * img_width)
    x_max_absolute = int(x_max * img_width)
    y_min_absolute = int(y_min * img_height)
    y_max_absolute = int(y_max * img_height)
    return (x_min_absolute, y_min_absolute), (x_max_absolute, y_max_absolute)

def abs_to_rel_coordinates(x_min: int, y_min: int, 
                           x_max: int, y_max: int, 
                           img_size: Tuple[int, int]
                           ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Convert absolute to relative rectangle coordinates        
    Parameter
    -------
    x_min : int, y_min : int, x_max : int, y_max : int
        Top-left and right-bottom absolute coordinates of rectangle
     
    img_size : Tuple[int, int]
        Image size (height, width)
    
    Returns
    -------
    Tuple[Tuple[float, float], Tuple[float, float]]
        Rectangle relative coordinates in format ((x_min, y_min), (x_max, y_max))
    """
    img_height, img_width = img_size

    x_min_relative = x_min / img_width
    x_max_relative = x_max / img_width
    y_min_relative = y_min / img_height
    y_max_relative = y_max / img_height
    return (x_min_relative, y_min_relative), (x_max_relative, y_max_relative)

def tlwh_to_tlbr(x_min, y_min, w, h):
    return (x_min, y_min), (x_min + w, y_min + h)

def visualize_detections(img: np.ndarray, detections: List[Detect]) -> np.ndarray:
    """
    Draw bbox and add text with class name and confidence     
    Parameter
    -------
    img : np.ndarray
        Image in BGR format with shape of (height, width, channels)
     
    detections: List[Detect]
        List of unified detections for current image  
    
    Returns
    -------
    np.ndarray
        Reshaped image with visualuzed detection, class name and confidence. Shape: (height, width, channels)
    """
    new_img = img.copy()
    for detection in detections:
        top_left, bottom_right = rel_to_abs_coordinates(detection.bbox.x_min, 
                                                        detection.bbox.y_min, 
                                                        detection.bbox.x_max, 
                                                        detection.bbox.y_max, 
                                                        img_size = OUTPUT_IMAGE_SIZE)
        color = (0,0,255) # Red in BGR
        text = f'{detection.class_name}: {detection.conf:.3f}'
        text_shift = 5
        new_img = cv2.resize(new_img, OUTPUT_IMAGE_SIZE)
        new_img = cv2.rectangle(new_img, 
                                pt1=top_left, 
                                pt2=bottom_right, 
                                color=color, 
                                thickness=2)
        new_img = cv2.putText(img=new_img, 
                              text=text, 
                              org=(top_left[0], top_left[1] - text_shift), 
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale=0.6, 
                              color=color, 
                              thickness=2)
    return new_img


def print_timing_table(timings: Timings, images_count: int, model_name : str) -> None:
    timing_table = tabulate([
        ['-',                'Time per image, sec',                       'Total time, sec',             'Percents, %'],
        ['Preprocess time',   timings.preprocess_time / images_count,      timings.preprocess_time,       round(timings.percents()[0] * 100)],
        ['Inference time',    timings.inference_time / images_count,       timings.inference_time,        round(timings.percents()[1] * 100)],
        ['Postprocess time',  timings.postprocess_time / images_count,     timings.postprocess_time,      round(timings.percents()[2] * 100)],
        ['Total time',        timings.total_time() / images_count,         timings.total_time(),         '100'],
    ], headers='firstrow', tablefmt='fancy_grid')
    
    print(f'Timings for {model_name}:\n{timing_table}')