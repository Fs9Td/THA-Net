"""Persistent Activity Context Modeling (PACM)。"""

import torch
import torch.nn as nn


class PersistentActivityContextModeling(nn.Module):
    def __init__(self, channels, spatial_shape):
        super().__init__()
        self.channels = int(channels)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.activity_head = nn.Sequential(
            nn.Linear(channels + 1, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, features, coords, trace_gate):
        batch_size = int(coords[:, 0].max().item()) + 1 if coords.numel() else 1
        size_x, size_y = self.spatial_shape[:2]
        # xy key 保留 batch 维度，保证 DDP 中每个样本的空间图相互独立。
        xy_keys = (coords[:, 0].long() * size_x + coords[:, 1].long()) * size_y + coords[:, 2].long()
        unique_keys, inverse = torch.unique(xy_keys, sorted=True, return_inverse=True)
        counts = torch.bincount(inverse, minlength=unique_keys.numel()).float()
        activity = torch.log1p(counts)[inverse] / torch.log(features.new_tensor(float(max(self.spatial_shape[2], 2)), dtype=torch.float32))
        kappa = (1.0 - trace_gate.squeeze(1).float()).clamp(0.0, 1.0) * torch.sigmoid(
            self.activity_head(torch.cat((features, activity[:, None]), dim=1))
        ).squeeze(1).float()

        # AMP 下 gate/head 可能保持 FP32，而 sparse feature 为 FP16；使用
        # FP32 累加可避免 index_add_ 类型冲突和长时间池化的舍入误差。
        pooled = torch.zeros(
            (unique_keys.numel(), self.channels), device=features.device, dtype=torch.float32
        )
        pooled.index_add_(0, inverse, kappa[:, None] * features.float())
        denominator = torch.zeros((unique_keys.numel(),), device=features.device, dtype=torch.float32)
        denominator.index_add_(0, inverse, kappa)
        pooled = (pooled / denominator.clamp_min(1e-5)[:, None]).to(features.dtype)

        spatial_area = size_x * size_y
        flat_map = features.new_zeros((batch_size * spatial_area, self.channels))
        flat_map[unique_keys] = pooled
        activity_map = flat_map.view(batch_size, size_x, size_y, self.channels).permute(0, 3, 1, 2).contiguous()
        refined = self.refine(activity_map)
        prior_map = refined.permute(0, 2, 3, 1).reshape(batch_size * spatial_area, self.channels)
        prior = prior_map[xy_keys]
        return {
            "prior": prior,
            "activity": activity,
            "kappa": kappa,
            "activity_map": refined,
        }
