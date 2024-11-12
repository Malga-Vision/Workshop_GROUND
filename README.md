# Code for GROUND Workshop 2024 - Accepted Poster
## Proof of Concept in Anticipation using Head Direction Cues

<p style="text-align:center;"><a href="https://doi.org/10.48550/arXiv.2408.05516">Link to Paper</a></p>

Folder Structure
```
Workshop_GROUND/
├── example_dir/
│   ├── 00_drinking_0.txt
│   └── README.md
├── README.md
├── bboxes_load.py
├── coco_classes.py
├── env_spec.txt
├── env_spec.yaml
├── evn_cv2_spec.yaml
├── hpe_load.py
├── kpts_data_base_load.py
├── main.py
├── main_video.py
├── scratch.py
└── supervision_code.py
```

There are two main environments to be used 

``` main_video.py``` is the starting file to use YOLO in inference to extract bounding boxes and detections of objects.

pip install ultralytics -> version 8.2.31
uninstall opencv-python
Using opencv-python-headless
due to https://github.com/opencv/opencv-python/issues/386