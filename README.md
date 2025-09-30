# Parameter-Free Logit Distillation via Sorting Mechanism [IEEE SPL'25]
[![](https://img.shields.io/badge/arXiv-2508.16544-green.svg)](https://arxiv.org/abs/2508.16544) [![](https://img.shields.io/badge/paper-ieee.spl-red.svg)](https://ieeexplore.ieee.org/document/11141360/)

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
  
For KD:
  ```bash
  python3 tools/train.py --cfg configs/cifar100/kd.yaml
  ```
For DKD:
  ```bash
  python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml
  ```

2. Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`.
  
For KD:
  ```bash
  python3 tools/train.py --cfg configs/imagenet/r34_r18/kd.yaml
  ```
For DKD:
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

This work is based on [mdistiller](<https://github.com/megvii-research/mdistiller>). 
Sincere gratitude to the contributors of mdistiller, CTKD, and LSKD for their amazing efforts.

