#!/usr/bin/env bash
# Install all Python dependencies required by DeblurGANv2.
# Mirrors the install cells from demo_train.ipynb / demo_infer.ipynb.
set -e

pip install -U pip wheel
pip install "setuptools<70" appdirs
pip install torch torchvision torchsummary pretrainedmodels numpy \
    opencv-python-headless joblib albumentations tqdm tensorboardx fire
pip install scikit-image
