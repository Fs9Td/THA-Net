# THA-Net: Motion-Aligned Sparse Trace Recovery with Persistence-Gated Decoding for Long-Window Event Stream Segmentation


## Requirements

- Python 3.8.20
- PyTorch 1.9.1+cu111
- CUDA runtime 11.1
- spconv-cu113 2.3.6
- NumPy 1.24.1
- SciPy 1.10.1
- PyYAML 6.0.2



## Installation

```bash
conda create -n tha-net python=3.8
conda activate tha-net
pip install -r requirements.txt
```

## Dataset

EV-UAV data are not included. Download or prepare EV-UAV yourself, preserve the official split, and provide the dataset root either in the YAML configuration or through `--data-root`.

```text
/path/to/EV-UAV/
├── train/*.npz
├── val/*.npz
└── test/*.npz
```

Each `.npz` file must contain `ev_loc` and `evs_norm`, following the EV-UAV format used by the EV-SpSegNet-compatible pipeline. The default configuration uses `data.root: datasets/EV-UAV`.

## Training

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_thanet.py \
    --config configs/tha_ev_uav.yaml \
    --data-root /path/to/EV-UAV
```

Two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --standalone --nproc_per_node=2 \
    train_thanet.py --config configs/tha_ev_uav.yaml \
    --data-root /path/to/EV-UAV --distributed
```


## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python eval_thanet.py \
    --config configs/tha_ev_uav_final.yaml \
    --checkpoint checkpoints/tha_net_ev_uav_best.pth \
    --data-root /path/to/EV-UAV \
    --split test
```



## Results

| Method   | IoU   | ACC   | Pd    | Fa (×10^-4) |

|  THA-Net | 65.21 | 72.53 | 86.74 | 0.30 |


Small numerical variation may occur with different GPUs, CUDA versions, or sparse-convolution builds.


## Acknowledgements

We thank the authors of [EV-UAV](https://github.com/ChenYichen9527/Ev-UAV) for open-sourcing their code, which provides a valuable foundation for this work.

