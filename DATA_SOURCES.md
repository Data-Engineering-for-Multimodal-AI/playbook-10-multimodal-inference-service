# Data licenses

This file lists the datasets and models this playbook uses, where they come from, and their licenses. The code in this repository is MIT-licensed. That license does not cover third-party data or models; each one keeps its own license, listed below.

## Datasets

| Dataset | Source | License | In this repo |
|---|---|---|---|
| RLBench `push_buttons` demonstrations (25, from the RLBench 18-task set, PerAct protocol, 128x128 cameras) | [huggingface.co/datasets/shihao1895/rlbench](https://huggingface.co/datasets/shihao1895/rlbench), file `test/push_buttons.tar.xz` at pinned revision `b44678905fb2`. Generated with [RLBench](https://github.com/stepjam/RLBench) | MIT, as tagged on the Hugging Face dataset. RLBench's software licence (Imperial College London) is for non-commercial use (see Notes) | No. Downloaded by `make data` or `make setup` (`python download_sample_data.py`) into `rlbench-data/push_buttons/`: a 349 MB download, of which about 78 MB is kept. The script checks the archive's SHA-256 |

## Models

| Model | Source | License | How it's used |
|---|---|---|---|
| ResNet-18, ImageNet weights (`ResNet18_Weights.IMAGENET1K_V1`) | [torchvision](https://pytorch.org/vision/stable/models.html) | BSD-3-Clause (torchvision), as the README states | Camera-frame backbone in `model.py`. Downloaded by torchvision on first use |

## Notes

- The Hugging Face dataset is tagged MIT. The repo says the demonstrations were generated with RLBench, whose software licence (Imperial College London) is for non-commercial use. Check both before any commercial use of the data.

## Citations

- RLBench: S. James, Z. Ma, D. Rovick Arrojo and A. J. Davison, "RLBench: The Robot Learning Benchmark & Learning Environment", *IEEE Robotics and Automation Letters*, 2020.
- ResNet: K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition", arXiv:1512.03385, 2015.
