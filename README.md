# model-speed-estimator

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
├── datasets.zip
├── download_models.sh
├── models
├── poetry.lock
├── pyproject.toml
├── run.py
├── utils
└── vendor
```


# Install
```bash
poetry install
poetry shell
```
#TODO:
1. rename directory "vendor"
1. Add tests
1. Add types
