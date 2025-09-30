# Parameter-Free Logit Distillation via Sorting Mechanism [IEEE SPL'25]
[![](https://img.shields.io/badge/arXiv-2508.16544-green.svg)](https://arxiv.org/abs/2508.16544) [![](https://img.shields.io/badge/paper-ieee.spl-red.svg)](https://ieeexplore.ieee.org/document/11141360/)

### Main Benchmark Results

On CIFAR-100:


| Teacher <br> Student |ResNet56 <br> ResNet20|ResNet110 <br> ResNet32| ResNet32x4 <br> ResNet8x4| WRN-40-2 <br> WRN-16-2| WRN-40-2 <br> WRN-40-1 | VGG13 <br> VGG8|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|
| KD | 70.66 | 73.08 | 73.33 | 74.92 | 73.54 | 72.98 |
| **DKD** | **71.97** | **74.11** | **76.32** | **76.23** | **74.81** | **74.68** |


| Teacher <br> Student |ResNet32x4 <br> ShuffleNet-V1|WRN-40-2 <br> ShuffleNet-V1| VGG13 <br> MobileNet-V2| ResNet50 <br> MobileNet-V2| ResNet32x4 <br> MobileNet-V2|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|
| KD | 74.07 | 74.83 | 67.37 | 67.35 | 74.45 |
| **DKD** | **76.45** | **76.70** | **69.71** | **70.35** | **77.07** |


On ImageNet:

| Teacher <br> Student |ResNet34 <br> ResNet18|ResNet50 <br> MobileNet-V1|
|:---------------:|:-----------------:|:-----------------:|
| KD | 71.03 | 70.50 | 
| **DKD** | **71.70** | **72.05** |

### Installation

Environments:

- Python 3.6
- PyTorch 1.9.0
- torchvision 0.10.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```

### Getting started

0. Wandb as the logger

- The registration: <https://wandb.ai/home>.
- If you don't want wandb as your logger, set `CFG.LOG.WANDB` as `False` at `mdistiller/engine/cfg.py`.

1. Training on CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.
  
### KD:
  ```bash
  python3 tools/train.py --cfg configs/cifar100/kd.yaml
  ```
### DKD:
  ```bash
  python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml
  ```

2. Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`.
  
### KD:
  ```bash
  python3 tools/train.py --cfg configs/imagenet/r34_r18/kd.yaml
  ```
### DKD:
  ```bash
  python3 tools/train.py --cfg configs/imagenet/r34_r18/dkd.yaml
  ```

# Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX
@article{sortkd25,
  title={Parameter-Free Logit Distillation via Sorting Mechanism},
  author={Limantoro, Stephen Ekaputra},
  journal={IEEE Signal Processing Letters},
  year={2025},
  publisher={IEEE}
}
```

# License

MDistiller is released under the MIT license. See [LICENSE](LICENSE) for details.

# Acknowledgement

This work is based on mdistiller. 
Sincere gratitude to the contributors of mdistiller, CTKD, and LSKD for their distinguished efforts.

