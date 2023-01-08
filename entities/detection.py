from dataclasses import dataclass

@dataclass
class BBox:
    """BBox with relative coordinates
    """
    x_min: float 
    y_min: float 
    x_max: float 
    y_max: float


@dataclass
class Detect:
    bbox: BBox
    conf: float
    label: int
    class_name: str