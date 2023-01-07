# TODO Terminate script if error

omz_downloader --output_dir vendor/models --name person-detection-0200

mkdir vendor/models/yolov4-tiny
wget -P vendor/models/yolov4-tiny https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget -P vendor/models/yolov4-tiny https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
