# Lungs-Tumor-segmentation
This project aims to segment lungs tumor from CT scans of Decathlon Lungs dataset using Pytorch and MONAI .

> This project is ***still under development***.
## Directory structure
```bash
app/
├── main.py          - FastAPI app
├── gunicorn.py      - WSGI script
└── schemas.py       - API model schemas

config/
└── config.py        - configuration setup
 
segmentor/
├── datasets/        - datasets
├── data.py          - data processing components
├── eval.py          - evaluation components
├── main.py          - training/optimization pipelines
├── models.py        - model architectures
├── predict.py       - inference components
├── train.py         - training components
└── utils.py         - supplementary utilities
```
## Workflows

1. Set up environment.
```bash
python -m venv venv         # create environment
source venv/bin/activate    # activate environment 
```

## Acknowledgements

 - [Pytorch](https://pytorch.org/docs/stable/index.html)
 - [MONAI](https://monai.io/index.html)
 - [Made With ML](https://madewithml.com/)
 - [Preprocessing 3D Volumes for Tumor Segmentation Using Monai and PyTorch](https://pycad.co/preprocessing-3d-volumes-for-tumor-segmentation-using-monai-and-pytorch/)


## Authors

- [@bhimrazy](https://www.github.com/bhimrazy)


## License

[MIT](https://github.com/bhimrazy/Lungs-Tumor-segmentation/blob/main/LICENSE)

