# model-speed-estimator
The Model Speed Estimator can help you to estimate preprocess, inference and postprocess time for different models.
## Input formats
* images (.jpg, .jpeg)
* videos (.mp4) 

*NOTE: The Model Speed Estimator may work fine with other formats of video and image but it's not a guarantee*
# Prepare project directory
1. Download test datasets [here](https://drive.google.com/file/d/1l7oX-q4zZMBRhpu2ayciQDFIpqzcLZ0r/view?usp=sharing)
1. Unzip datasets in a way to get such structure:
```
<project_root>
├── README.md
├── datasets
│   ├── images
│   │   ├── people_0.jpg
│   │   ├── people_1.jpg
│   │   ├── ...
│   └── videos
│       ├── people_05s.mp4
│       ├── people_10s.mp4
│       ├── ...
├── download_models.sh
├── models
├── poetry.lock
├── pyproject.toml
├── run.py
├── utils
└── vendor
```


# Install and setup environment
```bash
# Use python 3.8
poetry install
./download_models.sh
```

# Run demo
```bash
# Run image processing
poetry run python run.py --img-dir-path datasets/images

# Run video processing
poetry run python run.py --video-path datasets/videos/people_20s.mp4
```

# TODO:
1. Add tests
