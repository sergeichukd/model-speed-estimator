from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any
from entities.detection import Detect


class Model(ABC):
    name: str

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def infer(self):
        pass

    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> Any:
        pass

    @abstractmethod
    def unify_prediction(self, predictions: Any, img_size: Any) -> List[Detect]:
        pass
