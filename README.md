# Lungs-Tumor-segmentation

This project aims to segment lungs tumor from CT scans of Decathlon Lungs dataset using Pytorch and MONAI .

> This project is **_still under development_**.

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
├── notebook.ipynb   - notebook file
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

## References

- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [MONAI](https://monai.io/index.html)
- [LightningAI](https://lightning.ai/)
- [Brain Tumor Segmentation using MONAI and WandB](https://wandb.ai/geekyrakshit/brain-tumor-segmentation/reports/Brain-Tumor-Segmentation-using-MONAI-and-WandB---Vmlldzo0MjUzODIw)
- https://github.com/Project-MONAI/tutorials/tree/main/3d_segmentation

## Authors

- [@bhimrazy](https://www.github.com/bhimrazy)

## License

[MIT](./LICENSE)
