from openvino.runtime import Core
from pathlib import Path


MODEL_DIR = Path('vendor/models/intel/person-detection-0200')

class PersonDetection0200:
    # MAKE proper name for "accuracy" and "device" variable
    def __init__(self, accuracy='FP32', device: str='CPU', name='person_detection_0200') -> None:
        ie = Core()
        model_path = MODEL_DIR / accuracy / 'person-detection-0200.xml'
        self.model = ie.read_model(model=model_path.as_posix())
        self.compiled_model = ie.compile_model(model=self.model, device_name=device.upper())
        self.output_layer = self.compiled_model.output(0)
        self.name = name
    
    def infer(self, image):
        return self.compiled_model([image])[self.output_layer]
