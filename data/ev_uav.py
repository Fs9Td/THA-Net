"""EV-UAV 数据读取与稀疏体素聚合。

数据文件沿用 EV-SpSegNet 的格式：``ev_loc`` 是整数体素坐标，
``evs_norm`` 的前四列是归一化的 x、y、t、polarity，后两列分别是
事件标签和目标实例 id。
"""

import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class EVUAVDataset(Dataset):
    def __init__(self, root: str, split: str, max_events: int = 500000, seed: int = 37, augment: bool = False, resolution=(346, 260)):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.split = split
        self.max_events = int(max_events)
        self.seed = int(seed)
        self.augment = bool(augment and split == "train")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        split_root = os.path.join(self.root, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError("EV-UAV split 不存在: {}".format(split_root))
        self.files = [
            os.path.join(split_root, name)
            for name in sorted(os.listdir(split_root))
            if name.endswith(".npz")
        ]
        if not self.files:
            raise RuntimeError("EV-UAV split 为空: {}".format(split_root))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        with np.load(self.files[index]) as data:
            locations = np.asarray(data["ev_loc"], dtype=np.int64)
            normalized = np.asarray(data["evs_norm"], dtype=np.float32)

        features = normalized[:, :4]
        labels = normalized[:, 4].astype(np.float32, copy=False)
        instance_ids = normalized[:, 5].astype(np.int64, copy=False)

        if self.max_events > 0 and len(locations) > self.max_events:
            # 论文约束每个窗口最多 5e5 events。训练集随机抽样增加扰动；
            # val/test 使用均匀时间覆盖，避免评测结果依赖一次随机抽样。
            if self.split == "train":
                rng = np.random.default_rng(self.seed + index + len(locations))
                keep = rng.choice(len(locations), self.max_events, replace=False)
                keep.sort()
            else:
                keep = np.linspace(0, len(locations) - 1, self.max_events, dtype=np.int64)
            locations = locations[keep]
            features = features[keep]
            labels = labels[keep]
            instance_ids = instance_ids[keep]

        if self.augment:
            # 只做空间镜像，保留时间顺序和运动方向；归一化坐标使用
            # EV-SpSegNet 的 padded voxel 尺寸 352x288。
            rng = np.random.default_rng(self.seed * 1009 + index)
            padded_width = float(int(np.ceil(self.resolution[0] / 32.0) * 32))
            padded_height = float(int(np.ceil(self.resolution[1] / 32.0) * 32))
            if rng.random() < 0.5:
                locations[:, 0] = self.resolution[0] - 1 - locations[:, 0]
                features[:, 0] = locations[:, 0].astype(np.float32) / padded_width
            if rng.random() < 0.5:
                locations[:, 1] = self.resolution[1] - 1 - locations[:, 1]
                features[:, 1] = locations[:, 1].astype(np.float32) / padded_height

        return {
            "locations": locations,
            "features": features,
            "labels": labels,
            "instance_ids": instance_ids,
            "name": os.path.basename(self.files[index]),
        }


def _voxel_mean(coords: torch.Tensor, features: torch.Tensor):
    """在 CPU 上用 ``unique + index_add`` 做可移植的 mean voxelization。"""
    unique_coords, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
    voxel_features = torch.zeros(
        (unique_coords.shape[0], features.shape[1]), dtype=features.dtype
    )
    voxel_features.index_add_(0, inverse, features)
    counts = torch.bincount(inverse, minlength=unique_coords.shape[0]).to(features.dtype)
    voxel_features.div_(counts.clamp_min(1).unsqueeze(1))
    return unique_coords.contiguous(), voxel_features.contiguous(), inverse.contiguous()


def collate_ev_uav(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    """拼接样本并生成 event-to-voxel 映射。

    稀疏张量在训练进程中、搬到 GPU 后再构造，因而该 collate 函数可安全地
    配合多 worker 和 DDP 使用。
    """
    all_coords = []
    all_features = []
    all_labels = []
    all_instance_ids = []
    names = []
    for batch_index, sample in enumerate(batch):
        locations = torch.from_numpy(sample["locations"])
        batch_column = torch.full((locations.shape[0], 1), batch_index, dtype=torch.long)
        all_coords.append(torch.cat((batch_column, locations), dim=1))
        all_features.append(torch.from_numpy(sample["features"]))
        all_labels.append(torch.from_numpy(sample["labels"]))
        all_instance_ids.append(torch.from_numpy(sample["instance_ids"]))
        names.append(sample["name"])

    event_coords = torch.cat(all_coords, dim=0).contiguous()
    event_features = torch.cat(all_features, dim=0).float().contiguous()
    labels = torch.cat(all_labels, dim=0).float().contiguous()
    instance_ids = torch.cat(all_instance_ids, dim=0).long().contiguous()
    voxel_coords, voxel_features, event_to_voxel = _voxel_mean(event_coords, event_features)

    return {
        "voxel_coords": voxel_coords,
        "voxel_features": voxel_features,
        "event_to_voxel": event_to_voxel.long(),
        "event_coords": event_coords,
        "labels": labels,
        "instance_ids": instance_ids,
        "batch_size": torch.tensor(len(batch), dtype=torch.long),
        "names": names,
    }
