"""Persistence-Gated Trace Decoding (PGTD)。"""

import torch
import torch.nn as nn


class PersistenceGatedTraceDecoding(nn.Module):
    def __init__(self, channels, hidden=64, persistence_exponent=1.5, residual=False):
        super().__init__()
        self.persistence_exponent = float(persistence_exponent)
        self.residual = bool(residual)
        self.trace_score = nn.Sequential(
            nn.Linear(channels * 2 + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.persistence_score = nn.Sequential(
            nn.Linear(channels * 2 + 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.decode = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        # 将 motion-aligned trace response 注入 decoder feature，使 PGTD 的
        # trace response 分支和最终事件级 head 端到端相连。
        self.trace_inject = nn.Linear(1, channels, bias=False)
        if self.residual:
            nn.init.zeros_(self.decode[-1].weight)
            nn.init.zeros_(self.decode[-1].bias)
            nn.init.zeros_(self.trace_inject.weight)

    def forward(self, features, trace_descriptor, trace_gate, prior, activity):
        trace_score = self.trace_score(torch.cat((features, trace_descriptor, trace_gate), dim=1)).squeeze(1)
        persistence = torch.sigmoid(
            self.persistence_score(
                torch.cat((features, prior, 1.0 - trace_gate, activity[:, None]), dim=1)
            )
        ).squeeze(1)
        eta = (1.0 - persistence.clamp(0.0, 1.0).pow(self.persistence_exponent)).clamp(0.02, 1.0)
        probability = torch.sigmoid(trace_score) * eta
        decoded = self.decode(
            torch.cat((features, eta[:, None] * trace_descriptor, persistence[:, None] * prior), dim=1)
        )
        if self.residual:
            decoded = features + decoded
        decoded = decoded + self.trace_inject(trace_score[:, None])
        return {
            "features": decoded,
            "probability": probability[:, None],
            "trace_score": trace_score[:, None],
            "persistence": persistence[:, None],
            "reliability": eta[:, None],
        }
