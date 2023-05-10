# e6691-2023spring-project-person_removal
  - Welcome to the repository for e6691-2023spring-project-person_removal. The group is with members xb2165 (Xin Bu), yl5086 (Yufan Luo) and yj2732 (Yuliang Jiang).
  - The topic of this group project is person removal inspired by YOLO, DETR and Stable Diffusion.

# Our design
  -  The problem is to detect and remove persons for designated RGB input image. To solve this problem, several pre-trained deep networks are concatonated to take an RGB image as input, and the output is the inpainted image of size 512x512x3. For testing, the images and the masks are readed and loaded from .png files.
  - The code for loading images and masks can be found in utils.dataset.py.
  - The code for model construction can be found in utils.pipeline.py.
  - The code for person detection can be found in utils.person_segmentation.py.
  - Other helper functions can be found in utils.misc.py.
  - Person removal is done in the Jupyter notebook demo.ipynb.

# Our model
  - Our model detects person with DETR, generate masks by SAM, generate prompt by vit-gpt2, and inpaint the image with Stable Diffusion.
  - The model is built with PyTorch 2.0.0.

# Our result
  - Sample input

![My Image](demo_original.jpg)
  - Sample output

![My Image](demo_removed.jpg)

# Requirements
The libraries used for this project:

torch, torchvision, pillow, transformers, [segment_anything](https://github.com/facebookresearch/segment-anything), diffusers.

# Organization of this directory
```
./
├── README.md
├── demo.ipynb
├── demo_original.jpg
├── demo_removed.jpg
├── dataset
│   ├── COCO
│   └── PennFudanPed
├── images
│   ├── IMG_2508.jpg
│   ├── IMG_2510.jpg
│   ├── IMG_2511.jpg
│   ├── IMG_2517.jpg
│   ├── IMG_2518.jpg
│   ├── IMG_2521.jpg
│   └── README.md
├── model_ckpt
│   └── sam_vit_h_4b8939.pth
└── utils
    ├── dataset.py
    ├── misc.py
    ├── person_segmentation.py
    └── pipeline.py

5 directories, 16 files
```
