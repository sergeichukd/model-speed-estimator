import time
import distutils.dir_util
from pathlib import Path
from tqdm import tqdm
from typing import Union

from models.abstract_model import Model
from models.person_detection_0200 import PersonDetection0200
from models.yolov4_tiny import YoloV4Tiny
from utils.input_utils import ImageReader, VideoReader
from utils.cli_utils import parse_args
from utils.output_utils import visualize_detections, \
    ImageWriter, VideoWriter, Timings, print_timing_table, OUTPUT_IMAGE_SIZE


def estimate_model(model: Model,
                   data_path: Union[Path, str],
                   results_path: Union[Path, str],
                   is_process_video: bool):
    
    data_path = Path(data_path)
    model_results_path = Path(results_path) / model.name
    distutils.dir_util.mkpath(model_results_path.as_posix())
    
    timings = Timings()

    if is_process_video:
        data_reader = VideoReader(data_path)
        out_video_path = model_results_path / data_path.with_suffix('.avi').name
        writer = VideoWriter(out_video_path, 
                            fps=data_reader.fps, 
                            frame_size=OUTPUT_IMAGE_SIZE)
    else:
        data_reader = ImageReader(data_path)
        writer = ImageWriter(model_results_path)

    for orig_img in tqdm(data_reader):
        # Preprocess
        start_time = time.time()
        input_img = model.preprocess_image(orig_img)
        timings.preprocess_time += time.time() - start_time
        
        # Inference
        start_time = time.time()
        preds = model.infer(input_img)
        timings.inference_time += time.time() - start_time
        
        # Postprocess
        start_time = time.time()
        dets = model.unify_prediction(preds, input_img.shape[:2])
        vis_img = visualize_detections(orig_img, dets)
        writer.write(vis_img)
        timings.postprocess_time += time.time() - start_time
    writer.close()
    print_timing_table(timings, len(data_reader), model.name)
    print('FPS:', len(data_reader) / timings.total_time())

def main():
    args = parse_args()
    assert bool(args.video_path) != bool(args.img_dir_path), f'You should set up whether video path or image dir path'
    data_path = Path(args.video_path or args.img_dir_path)
    is_process_video = bool(args.video_path)

    if is_process_video:
        results_path = Path(f'results/videos_{time.strftime("%d-%m-%Y_%H-%M-%S")}')
    else:
        results_path = Path(f'results/images_{time.strftime("%d-%m-%Y_%H-%M-%S")}')

    #  Estimate person_detection_0200
    model = PersonDetection0200()
    estimate_model(model, data_path, results_path, is_process_video)

    # Estimate yolov4_tiny
    model = YoloV4Tiny()
    estimate_model(model, data_path, results_path, is_process_video)

if __name__ == '__main__':
    main()
