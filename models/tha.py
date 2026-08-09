"""Trace Hypothesis Aggregation (THA)。"""

import math

import torch
import torch.nn as nn
import spconv.pytorch as spconv


def _sparse_keys(coords, spatial_shape):
    """把 [batch, x, y, t] 坐标编码成可在 GPU 上查找的整数 key。"""
    shape_x, shape_y, shape_t = [int(v) for v in spatial_shape]
    coords = coords.long()
    return (((coords[:, 0] * shape_x + coords[:, 1]) * shape_y + coords[:, 2]) * shape_t + coords[:, 3])


def sparse_nearest_lookup(query_coords, source_coords, source_features, spatial_shape):
    """对稀疏坐标做向量化的 exact-nearest voxel lookup。

    THA 的采样坐标先舍入到当前稀疏尺度的 voxel 网格；不存在的坐标被标成
    invalid，调用方在聚合时将其权重置零。这与论文中的 nearest sparse lookup
    和 invalid mask 相对应。
    """
    source_keys = _sparse_keys(source_coords, spatial_shape)
    sorted_keys, order = torch.sort(source_keys)
    query_keys = _sparse_keys(query_coords, spatial_shape)
    positions = torch.searchsorted(sorted_keys, query_keys)
    valid = positions < sorted_keys.numel()
    safe_positions = positions.clamp_max(max(sorted_keys.numel() - 1, 0))
    valid = valid & (sorted_keys[safe_positions] == query_keys)
    output = source_features.new_zeros((query_coords.shape[0], source_features.shape[1]))
    if source_features.shape[0] > 0:
        output[valid] = source_features[order[safe_positions[valid]]]
    return output, valid


def sparse_lookup_indices(query_coords, source_coords, spatial_shape, radius=0):
    """返回 query 对应的 source index；radius>0 时在邻近稀疏网格搜索。"""
    source_keys = _sparse_keys(source_coords, spatial_shape)
    sorted_keys, order = torch.sort(source_keys)
    source_index = torch.full(
        (query_coords.shape[0],), -1, dtype=torch.long, device=query_coords.device
    )
    if source_keys.numel() == 0 or query_coords.numel() == 0:
        return source_index

    # 半径搜索的候选数为 (2r+1)^3；EV-UAV 达到 5e5 events 时若一次展开
    # 全部 query 会产生数十 GB 临时张量。分块后显存开销与事件数解耦。
    chunk_size = 8192
    axis = torch.arange(-radius, radius + 1, device=query_coords.device, dtype=query_coords.dtype)
    offsets = torch.cartesian_prod(axis, axis, axis) if radius > 0 else query_coords.new_zeros((1, 3))
    for start in range(0, query_coords.shape[0], chunk_size):
        end = min(start + chunk_size, query_coords.shape[0])
        chunk = query_coords[start:end]
        candidates = chunk[:, None, :].expand(-1, offsets.shape[0], -1).clone()
        if radius > 0:
            candidates[:, :, 1:] += offsets[None, :, :]
        query_keys = _sparse_keys(candidates.reshape(-1, 4), spatial_shape)
        positions = torch.searchsorted(sorted_keys, query_keys)
        valid = positions < sorted_keys.numel()
        safe_positions = positions.clamp_max(sorted_keys.numel() - 1)
        valid = valid & (sorted_keys[safe_positions] == query_keys)
        valid = valid.view(candidates.shape[0], candidates.shape[1])
        safe_positions = safe_positions.view(candidates.shape[0], candidates.shape[1])
        found = valid.any(dim=1)
        first = valid.float().argmax(dim=1)
        target = source_index[start:end]
        target[found] = order[safe_positions[found, first[found]]]
    return source_index


class TraceHypothesisAggregation(nn.Module):
    def __init__(
        self,
        channels,
        spatial_shape,
        hypotheses=4,
        offsets=(-4, -2, 0, 2, 4),
        max_motion=2.0,
        temporal_penalty=0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.hypotheses = int(hypotheses)
        self.offsets = tuple(int(v) for v in offsets)
        self.max_motion = float(max_motion)
        self.temporal_penalty = float(temporal_penalty)
        coord_dim = 8
        self.coord_embed = nn.Sequential(nn.Linear(3, coord_dim), nn.ReLU(inplace=True))
        self.motion_head = nn.Linear(channels + coord_dim, hypotheses * 3)
        self.query_proj = nn.Linear(channels, channels, bias=False)
        self.sample_proj = nn.Linear(channels, channels, bias=False)
        self.value_proj = nn.Linear(channels, channels, bias=False)
        self.trace_gate = nn.Sequential(
            nn.Linear(channels * 2 + coord_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )
        self.trace_out = nn.Linear(channels, channels, bias=False)
        # 以已有稀疏 U-Net/随机初始化的主干为起点时，THA 初始应是残差零映射；
        # 训练后再逐步学习有效的跨时间 trace，避免随机 trace 破坏预训练解码器。
        nn.init.zeros_(self.trace_out.weight)

    def forward(self, sparse_tensor):
        features = sparse_tensor.features
        coords = sparse_tensor.indices.long()
        n = features.shape[0]
        if n == 0:
            raise RuntimeError("THA 收到空 sparse tensor")

        norm = features.new_tensor(
            [max(self.spatial_shape[0] - 1, 1), max(self.spatial_shape[1] - 1, 1), max(self.spatial_shape[2] - 1, 1)]
        )
        coord_embedding = self.coord_embed(coords[:, 1:].float() / norm)
        motion_raw = self.motion_head(torch.cat((features, coord_embedding), dim=1))
        motion_raw = motion_raw.view(n, self.hypotheses, 3)
        motion = self.max_motion * torch.tanh(motion_raw[:, :, :2])
        confidence = motion_raw[:, :, 2]

        offsets = features.new_tensor(self.offsets).view(1, 1, -1)
        query = coords.float().view(n, 1, 1, 4).expand(-1, self.hypotheses, len(self.offsets), -1).clone()
        query[:, :, :, 1] += offsets * motion[:, :, None, 0]
        query[:, :, :, 2] += offsets * motion[:, :, None, 1]
        query[:, :, :, 3] += offsets
        query = torch.round(query).long().view(-1, 4)
        sampled, valid = sparse_nearest_lookup(
            query, coords, features, sparse_tensor.spatial_shape
        )
        sampled = sampled.view(n, self.hypotheses, len(self.offsets), self.channels)
        valid = valid.view(n, self.hypotheses, len(self.offsets))

        q = self.query_proj(features)
        s = self.sample_proj(sampled)
        gamma = (q[:, None, None, :] * s).sum(dim=-1) / math.sqrt(self.channels)
        gamma = gamma - self.temporal_penalty * offsets.abs()
        gamma = gamma.masked_fill(~valid, -1e4)
        gamma_max = gamma.max(dim=2, keepdim=True).values
        weights = torch.exp((gamma - gamma_max).clamp(min=-40.0, max=20.0)) * valid.float()
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)

        reliability = confidence + (gamma * valid.float()).sum(dim=2) / valid.float().sum(dim=2).clamp_min(1.0)
        alpha = torch.softmax(reliability, dim=1)
        values = self.value_proj(sampled)
        trace_descriptor = (alpha[:, :, None, None] * weights[:, :, :, None] * values).sum(dim=(1, 2))
        trace_gate = torch.sigmoid(
            self.trace_gate(torch.cat((features, trace_descriptor, coord_embedding), dim=1))
        )
        updated = features + trace_gate * self.trace_out(trace_descriptor)
        return {
            "features": updated,
            "trace_descriptor": trace_descriptor,
            "trace_gate": trace_gate,
            "motion": motion,
            "alpha": alpha,
            "confidence": confidence,
            "coord_embedding": coord_embedding,
        }
