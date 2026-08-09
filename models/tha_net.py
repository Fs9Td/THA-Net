"""THA-Net：稀疏 3D U-Net + THA + PACM + PGTD。"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from .pacm import PersistentActivityContextModeling
from .pgtd import PersistenceGatedTraceDecoding
from .tha import TraceHypothesisAggregation, sparse_lookup_indices


def _norm(channels):
    return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)


class SparseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, indice_key):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = _norm(out_channels)
        self.conv2 = spconv.SubMConv3d(
            out_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = _norm(out_channels)
        self.skip = None
        if in_channels != out_channels:
            self.skip = spconv.SubMConv3d(
                in_channels, out_channels, 1, bias=False, indice_key=indice_key + "_skip"
            )
            self.skip_bn = _norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.features
        y = self.conv1(x)
        y = y.replace_feature(self.relu(self.bn1(y.features)))
        y = self.conv2(y)
        y = y.replace_feature(self.bn2(y.features))
        if self.skip is not None:
            skip = self.skip(x)
            identity = self.skip_bn(skip.features)
        return y.replace_feature(self.relu(y.features + identity))


def _aligned_features(query, source):
    """按坐标对齐 skip feature，避免依赖 sparse backend 的内部排序。"""
    q = query.indices.long()
    s = source.indices.long()
    shape = query.spatial_shape
    source_keys = (((s[:, 0] * int(shape[0]) + s[:, 1]) * int(shape[1]) + s[:, 2]) * int(shape[2]) + s[:, 3])
    query_keys = (((q[:, 0] * int(shape[0]) + q[:, 1]) * int(shape[1]) + q[:, 2]) * int(shape[2]) + q[:, 3])
    sorted_keys, order = torch.sort(source_keys)
    pos = torch.searchsorted(sorted_keys, query_keys)
    valid = pos < sorted_keys.numel()
    safe = pos.clamp_max(max(sorted_keys.numel() - 1, 0))
    valid = valid & (sorted_keys[safe] == query_keys)
    aligned = query.features.new_zeros((q.shape[0], source.features.shape[1]))
    aligned[valid] = source.features[order[safe[valid]]]
    return aligned


class THANet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        model_cfg = config["model"]
        data_cfg = config["data"]
        self.input_channels = int(model_cfg.get("input_channels", 4))
        base = int(model_cfg.get("base_channels", 12))
        resolution = data_cfg.get("resolution", [346, 260])
        window_ms = int(data_cfg.get("window_ms", 8000))
        # 与 EV-SpSegNet 一致，保留足够的空间余量并让三次下采样可整除。
        self.input_spatial_shape = (
            int(math.ceil(float(resolution[0]) / 32.0) * 32),
            int(math.ceil(float(resolution[1]) / 32.0) * 32),
            int(math.ceil(float(window_ms) / 32.0) * 32),
        )
        self.high_spatial_shape = (
            self.input_spatial_shape[0] // 8,
            self.input_spatial_shape[1] // 8,
            self.input_spatial_shape[2] // 64,
        )

        c1, c2, c3, c4 = base, base * 2, base * 4, base * 4
        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channels, c1, 3, padding=1, bias=False, indice_key="stem"),
            _norm(c1),
            nn.ReLU(inplace=True),
        )
        self.enc1 = SparseResidualBlock(c1, c1, "enc1")
        self.down2 = spconv.SparseSequential(
            spconv.SparseConv3d(c1, c2, 3, stride=(2, 2, 4), padding=1, bias=False, indice_key="down2"),
            _norm(c2),
            nn.ReLU(inplace=True),
        )
        self.enc2 = SparseResidualBlock(c2, c2, "enc2")
        self.down3 = spconv.SparseSequential(
            spconv.SparseConv3d(c2, c3, 3, stride=(2, 2, 4), padding=1, bias=False, indice_key="down3"),
            _norm(c3),
            nn.ReLU(inplace=True),
        )
        self.enc3 = SparseResidualBlock(c3, c3, "enc3")
        self.down4 = spconv.SparseSequential(
            spconv.SparseConv3d(c3, c4, 3, stride=(2, 2, 4), padding=1, bias=False, indice_key="down4"),
            _norm(c4),
            nn.ReLU(inplace=True),
        )
        self.enc4 = SparseResidualBlock(c4, c4, "enc4")

        self.tha = TraceHypothesisAggregation(
            c4,
            self.high_spatial_shape,
            hypotheses=int(model_cfg.get("motion_hypotheses", 4)),
            offsets=tuple(model_cfg.get("trace_offsets", [-4, -2, 0, 2, 4])),
            max_motion=float(model_cfg.get("max_motion", 2.0)),
            temporal_penalty=float(model_cfg.get("temporal_penalty", 0.05)),
        )
        self.pacm = PersistentActivityContextModeling(c4, self.high_spatial_shape)
        self.pgtd = PersistenceGatedTraceDecoding(
            c4,
            hidden=int(model_cfg.get("persistence_hidden", 64)),
            persistence_exponent=float(model_cfg.get("persistence_exponent", 1.5)),
        )
        # 论文式 PGTD 高层前景概率到事件级 head 的可控融合。默认 0 保持
        # 既有 checkpoint 的输出完全不变；新实验可用小权重微调。
        self.pgtd_output_weight = float(model_cfg.get("pgtd_output_weight", 0.0))
        # 可选的模型内 logit 融合。它在 head 的 logit 空间利用 PGTD，避免
        # 概率平均把原本超过 0.9 的正样本压回阈值以下。
        self.pgtd_logit_weight = float(model_cfg.get("pgtd_logit_weight", 0.0))
        # 为兼容旧 checkpoint 保留该字段；公开 release 的两个配置均固定为
        # 0.0，标准评测没有命令行入口可启用统一输出校准。
        self.output_logit_bias = float(model_cfg.get("output_logit_bias", 0.0))

        self.inv4 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(c4, c3, 3, indice_key="down4", bias=False),
            _norm(c3),
            nn.ReLU(inplace=True),
        )
        self.dec3 = SparseResidualBlock(c3, c3, "dec3")
        self.inv3 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(c3, c2, 3, indice_key="down3", bias=False),
            _norm(c2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = SparseResidualBlock(c2, c2, "dec2")
        self.inv2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(c2, c1, 3, indice_key="down2", bias=False),
            _norm(c1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = SparseResidualBlock(c1, c1, "dec1")
        # head 直接输出 logit。旧版本在 AMP 下先 Sigmoid、再 torch.logit，
        # 会让置信度 margin loss 经过半精度饱和点；去掉无参数的 Sigmoid
        # 不改变 checkpoint 参数形状，推理时仍显式生成同一个概率。
        self.head = nn.Sequential(nn.Linear(c1, c1), nn.ReLU(inplace=True), nn.Linear(c1, 1))

    @staticmethod
    def _merge(up, skip):
        aligned = _aligned_features(up, skip)
        return up.replace_feature(up.features + aligned)

    def forward(self, sparse_input):
        input_coords = sparse_input.indices.long()
        x1 = self.enc1(self.stem(sparse_input))
        x2 = self.enc2(self.down2(x1))
        x3 = self.enc3(self.down3(x2))
        x4 = self.enc4(self.down4(x3))

        tha = self.tha(x4)
        pacm = self.pacm(tha["features"], x4.indices.long(), tha["trace_gate"])
        pgtd = self.pgtd(
            tha["features"], tha["trace_descriptor"], tha["trace_gate"], pacm["prior"], pacm["activity"]
        )
        x4 = x4.replace_feature(pgtd["features"])

        u3 = self._merge(self.inv4(x4), x3)
        u3 = self.dec3(u3)
        u2 = self._merge(self.inv3(u3), x2)
        u2 = self.dec2(u2)
        u1 = self._merge(self.inv2(u2), x1)
        u1 = self.dec1(u1)
        head_logit = self.head(u1.features).float()
        probability = torch.sigmoid(head_logit)

        # 把输入 voxel 映射到 THA 所在尺度，供 persistence/motion 辅助损失使用。
        scale = input_coords.new_tensor([1, 8, 8, 64])
        coarse_input = torch.div(input_coords, scale, rounding_mode="floor")
        input_to_high = sparse_lookup_indices(
            coarse_input, x4.indices.long(), x4.spatial_shape, radius=2
        )
        pgtd_probability = pgtd["probability"]
        high_probability = None
        if self.pgtd_logit_weight > 0.0 or self.pgtd_output_weight > 0.0:
            high_probability = pgtd_probability.squeeze(1)[input_to_high.clamp_min(0)]
            high_probability = high_probability * (input_to_high >= 0).float() + 0.5 * (input_to_high < 0).float()
        if self.pgtd_logit_weight > 0.0:
            pgtd_logit = torch.logit(high_probability.float().clamp(1e-5, 1.0 - 1e-5))[:, None]
            voxel_logit = head_logit + self.pgtd_logit_weight * pgtd_logit
            probability = torch.sigmoid(voxel_logit)
        elif self.pgtd_output_weight > 0.0:
            blend = min(self.pgtd_output_weight, 1.0)
            probability = (1.0 - blend) * probability + blend * high_probability[:, None]
        if self.output_logit_bias != 0.0:
            probability = torch.sigmoid(
                torch.logit(probability.float().clamp(1e-5, 1.0 - 1e-5)) + self.output_logit_bias
            )
        # 无 PGTD 概率融合时直接保留 head logit，避免 sigmoid/logit 往返；
        # 融合或显式校准时则返回实际输出概率对应的 logit。
        if self.pgtd_logit_weight > 0.0 and self.output_logit_bias == 0.0:
            # 上面的 logit 融合已经是实际输出的 logit。
            pass
        elif self.pgtd_output_weight <= 0.0 and self.output_logit_bias == 0.0:
            voxel_logit = head_logit
        else:
            voxel_logit = torch.logit(probability.float().clamp(1e-5, 1.0 - 1e-5))

        return {
            "voxel_probability": probability,
            "voxel_logit": voxel_logit[:, None],
            "output_coords": u1.indices,
            "high_coords": x4.indices,
            "input_to_high": input_to_high,
            "motion": tha["motion"],
            "alpha": tha["alpha"],
            "trace_gate": tha["trace_gate"],
            "persistence": pgtd["persistence"],
            "activity": pacm["activity"][:, None],
            "reliability": pgtd["reliability"],
            "pgtd_probability": pgtd_probability,
        }
