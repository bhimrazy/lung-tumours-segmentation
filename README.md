# Lungs-Tumor-segmentation

This project aims to segment lungs tumor from CT scans of Decathlon Lungs dataset using Pytorch and MONAI .

> This project is **_still under development_**.

## Directory structure

```bash
├── LICENSE                    # License file
├── README.md                  # Readme file
├── app
│   ├── gunicorn.py            # Configuration for Gunicorn server
│   ├── main.py                # Main application file
│   └── schemas.py             # Schema definitions
├── artifacts
│   └── checkpoints            # Directory for storing model checkpoints
├── conf
│   └── config.yaml            # Configuration file in YAML format
├── data/                      # data dir
├── logs/                      # Log dir  
├── notebook.ipynb             # Jupyter notebook file
├── requirements.txt           # File listing required Python packages
├── src/
│   ├── data_module.py         # Module for data processing
│   ├── model.py               # Module containing the model definition
│   └── utils.py               # Utility functions module
├── download_data.sh           # Shell script for downloading data
└── train.py                   # Script for training the model
```

## Workflows

1. Set up environment.

```bash
python -m venv venv         # create environment
source venv/bin/activate    # activate environment
```

2. Train model
```
python train.py
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
