"""THA-Net 的分割、运动一致性与 persistence gate 辅助损失。"""

import torch
import torch.nn.functional as F


def _focal_probability_loss(probability, target, alpha=0.75, gamma=2.0):
    # 避免 GPU float16 flush-to-zero 让 log(1e-5) 变成 -inf。
    probability = probability.float().clamp(1e-4, 1.0 - 1e-4)
    target = target.float()
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (-alpha_t * (1.0 - pt).pow(gamma) * torch.log(pt)).mean()


def _dice_loss(probability, target):
    probability = probability.float()
    target = target.float()
    numerator = 2.0 * (probability * target).sum() + 1.0
    denominator = probability.sum() + target.sum() + 1.0
    return 1.0 - numerator / denominator


def _event_to_voxel_target(event_to_voxel, labels, count):
    sums = labels.new_zeros((count,))
    sums.index_add_(0, event_to_voxel, labels)
    counts = labels.new_zeros((count,))
    counts.index_add_(0, event_to_voxel, torch.ones_like(labels))
    return (sums / counts.clamp_min(1.0) > 0.5).float()


def _persistence_loss(output, event_to_voxel, labels, config):
    high_map = output["input_to_high"][event_to_voxel]
    valid = high_map >= 0
    if not valid.any():
        return labels.new_zeros(())
    high_count = output["persistence"].shape[0]
    high_labels = _event_to_voxel_target(high_map[valid], labels[valid], high_count)
    # AMP 下 float16 的 1-1e-5 仍可能舍入成 1，辅助项显式提升到 fp32。
    persistence = output["persistence"].squeeze(1).float().clamp(1e-4, 1.0 - 1e-4)
    foreground = high_labels > 0.5
    hard_background = (~foreground) & (output["activity"].squeeze(1) >= float(config["loss"].get("persistence_activity_threshold", 0.25)))
    terms = []
    if foreground.any():
        terms.append(-torch.log1p(-persistence[foreground]).mean())
    if hard_background.any():
        terms.append(-torch.log(persistence[hard_background]).mean())
    if not terms:
        return labels.new_zeros(())
    return torch.stack(terms).mean()


def _pgtd_probability_loss(output, event_to_voxel, labels, config):
    """监督 PGTD 的高层前景概率，避免 gate 分支只被间接使用。"""
    high_map = output["input_to_high"][event_to_voxel]
    valid = high_map >= 0
    if not valid.any() or "pgtd_probability" not in output:
        return labels.new_zeros(())
    high_count = output["pgtd_probability"].shape[0]
    high_labels = _event_to_voxel_target(high_map[valid], labels[valid], high_count)
    high_probability = output["pgtd_probability"].squeeze(1)
    focal = _focal_probability_loss(high_probability, high_labels)
    dice = _dice_loss(high_probability, high_labels)
    return focal + 0.5 * dice


def _confidence_margin_loss(output, event_to_voxel, labels, config):
    """训练原始 foreground logit 的置信度间隔，不改变推理阈值。"""
    if "voxel_logit" not in output:
        return labels.new_zeros(())
    logits = output["voxel_logit"].squeeze(1)[event_to_voxel]
    positive = labels > 0.5
    negative = ~positive
    terms = []
    positive_margin = float(config["loss"].get("positive_logit_margin", 2.1972246))
    negative_margin = float(config["loss"].get("negative_logit_margin", -2.0))
    if positive.any():
        terms.append(torch.nn.functional.softplus(positive_margin - logits[positive]).mean())
    if negative.any():
        negative_logits = logits[negative].reshape(-1)
        if negative_logits.numel() == 0:
            negative_logits = None
        if negative_logits is None:
            negative_term = logits.new_zeros(())
        else:
            hard_fraction = float(config["loss"].get("hard_negative_fraction", 0.05))
            hard_count = max(
                1,
                min(
                    negative_logits.numel(),
                    int(negative_logits.numel() * min(max(hard_fraction, 0.0), 1.0)),
                ),
            )
            hard_logits = torch.topk(negative_logits, hard_count, largest=True).values
            negative_term = torch.nn.functional.softplus(hard_logits - negative_margin).mean()
        terms.append(float(config["loss"].get("negative_margin_weight", 0.1)) * negative_term)
    if not terms:
        return labels.new_zeros(())
    return torch.stack(terms).sum()


def _confidence_ranking_loss(output, event_to_voxel, labels, config):
    """让前景事件 logit 排在最高分背景事件之上。"""
    if "voxel_logit" not in output:
        return labels.new_zeros(())
    logits = output["voxel_logit"].squeeze(1)[event_to_voxel]
    positive_logits = logits[labels > 0.5].reshape(-1)
    negative_logits = logits[labels <= 0.5].reshape(-1)
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        return labels.new_zeros(())
    fraction = float(config["loss"].get("hard_negative_fraction", 0.05))
    hard_count = max(1, min(negative_logits.numel(), int(negative_logits.numel() * min(max(fraction, 0.0), 1.0))))
    hard_negative = torch.topk(negative_logits, hard_count, largest=True).values
    pair_count = min(4096, positive_logits.numel(), hard_negative.numel())
    positive_index = torch.randperm(positive_logits.numel(), device=logits.device)[:pair_count]
    negative_index = torch.randperm(hard_negative.numel(), device=logits.device)[:pair_count]
    margin = float(config["loss"].get("ranking_margin", 1.0))
    return torch.nn.functional.softplus(
        hard_negative[negative_index] - positive_logits[positive_index] + margin
    ).mean()


def _motion_loss(output, event_to_voxel, event_coords, labels, config):
    high_map = output["input_to_high"][event_to_voxel]
    valid_event = high_map >= 0
    foreground = (labels > 0.5) & valid_event
    if foreground.sum() < 2:
        return labels.new_zeros(())

    motion = output["motion"]
    alpha = output["alpha"]
    predicted = (motion * alpha[:, :, None]).sum(dim=1)
    event_motion = predicted[high_map.clamp_min(0)]
    coords = event_coords.long()
    # 数据本身按时间排序；再做一次稳定排序便于构造局部前后景事件对。
    order = torch.argsort(coords[:, 0] * 1000000000 + coords[:, 3])
    order = order[foreground[order]]
    if order.numel() < 2:
        return labels.new_zeros(())
    prev = order[:-1]
    curr = order[1:]
    dt = coords[curr, 3].float() - coords[prev, 3].float()
    dx = coords[curr, 1].float() - coords[prev, 1].float()
    dy = coords[curr, 2].float() - coords[prev, 2].float()
    same_batch = coords[curr, 0] == coords[prev, 0]
    same_instance = True
    max_time = float(config["loss"].get("motion_pair_time", 256))
    max_radius = float(config["loss"].get("motion_pair_radius", 16))
    pair = same_batch & (dt > 0) & (dt <= max_time) & (dx.abs() <= max_radius) & (dy.abs() <= max_radius)
    if not pair.any():
        return labels.new_zeros(())
    # THA motion 的单位是高层 voxel / 高层 temporal step。
    scale_x, scale_t = 8.0, 64.0
    target_motion = torch.stack((dx * scale_t / (dt * scale_x), dy * scale_t / (dt * scale_x)), dim=1)
    target_motion = target_motion.clamp(-2.0, 2.0)
    weights = torch.exp(-dt / scale_t)
    residual = (event_motion[curr] - target_motion).abs().sum(dim=1)
    return (residual[pair] * weights[pair]).sum() / weights[pair].sum().clamp_min(1e-5)


def thanet_loss(output, batch, config):
    event_to_voxel = batch["event_to_voxel"]
    labels = batch["labels"].to(output["voxel_probability"].device, non_blocking=True)
    event_to_voxel = event_to_voxel.to(labels.device, non_blocking=True)
    probability = output["voxel_probability"].squeeze(1)[event_to_voxel]
    focal = _focal_probability_loss(
        probability,
        labels,
        alpha=float(config["loss"].get("focal_alpha", 0.75)),
        gamma=float(config["loss"].get("focal_gamma", 2.0)),
    )
    dice = _dice_loss(probability, labels)
    segmentation = focal + float(config["loss"].get("dice_weight", 0.5)) * dice
    event_coords = batch["event_coords"].to(labels.device, non_blocking=True)
    motion = _motion_loss(output, event_to_voxel, event_coords, labels, config)
    persistence = _persistence_loss(output, event_to_voxel, labels, config)
    pgtd = _pgtd_probability_loss(output, event_to_voxel, labels, config)
    confidence = _confidence_margin_loss(output, event_to_voxel, labels, config)
    ranking = _confidence_ranking_loss(output, event_to_voxel, labels, config)
    total = segmentation + float(config["loss"].get("motion_weight", 0.05)) * motion
    total = total + float(config["loss"].get("persistence_weight", 0.1)) * persistence
    total = total + float(config["loss"].get("pgtd_probability_weight", 0.0)) * pgtd
    total = total + float(config["loss"].get("confidence_margin_weight", 0.0)) * confidence
    total = total + float(config["loss"].get("confidence_ranking_weight", 0.0)) * ranking
    return total, {
        "loss": float(total.detach().item()),
        "seg": float(segmentation.detach().item()),
        "motion": float(motion.detach().item()),
        "persistence": float(persistence.detach().item()),
        "pgtd": float(pgtd.detach().item()),
        "confidence": float(confidence.detach().item()),
        "ranking": float(ranking.detach().item()),
    }
