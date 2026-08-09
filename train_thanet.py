"""THA-Net EV-UAV 训练入口，支持单卡与 torchrun DDP。"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import spconv.pytorch as spconv

from data import EVUAVDataset, collate_ev_uav
from models import THANet
from utils.config import load_config
from utils.losses import thanet_loss
from utils.metrics import EVUAVMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="训练 THA-Net on EV-UAV")
    parser.add_argument("--config", default="configs/tha_ev_uav.yaml")
    parser.add_argument("--data-root", default="", help="可选 EV-UAV 根目录，覆盖 config 的 data.root")
    parser.add_argument("--distributed", action="store_true", help="兼容显式 torchrun 命令；torchrun 本身会自动识别")
    parser.add_argument("--resume", default="", help="可选 latest checkpoint")
    parser.add_argument("--epochs", type=int, default=0, help="仅用于短 pilot；0 表示使用 config")
    parser.add_argument("--max-steps", type=int, default=0, help="每个 epoch 最多训练多少 step；0 表示不限制")
    parser.add_argument("--output-dir", default="", help="可选输出目录覆盖 config，便于 pilot")
    parser.add_argument("--no-amp", action="store_true", help="关闭 AMP，用于数值稳定性排查")
    parser.add_argument("--max-events", type=int, default=0, help="仅用于 pilot 覆盖每个训练样本的事件上限")
    parser.add_argument("--workers", type=int, default=-1, help="覆盖 DataLoader worker 数量")
    parser.add_argument("--init-checkpoint", default="", help="只加载模型权重，不恢复优化器，用于有方向微调")
    parser.add_argument("--learning-rate", type=float, default=0.0, help="覆盖 config 学习率")
    parser.add_argument("--motion-weight", type=float, default=-1.0, help="覆盖 motion auxiliary loss 权重")
    parser.add_argument("--dice-weight", type=float, default=-1.0, help="覆盖 Dice loss 权重")
    parser.add_argument("--focal-alpha", type=float, default=-1.0, help="覆盖 focal 正样本权重，用于 class imbalance")
    parser.add_argument("--augment", action="store_true", help="启用 train 空间镜像增强")
    parser.add_argument("--pgtd-output-weight", type=float, default=-1.0, help="PGTD 高层前景概率到事件 head 的融合权重")
    parser.add_argument("--pgtd-logit-weight", type=float, default=-1.0, help="PGTD 高层 logit 到事件 head 的模型内融合权重")
    parser.add_argument("--pgtd-probability-weight", type=float, default=-1.0, help="PGTD 高层前景概率监督损失权重")
    parser.add_argument("--freeze-backbone", action="store_true", help="冻结稀疏主干参数并固定其 BN 统计，仅训练 THA/PACM/PGTD")
    parser.add_argument("--train-confidence-head", action="store_true", help="冻结主干时额外训练最终 segmentation head")
    parser.add_argument("--head-only", action="store_true", help="仅训练 segmentation head；自动冻结 THA/PACM 和稀疏主干")
    parser.add_argument("--confidence-margin-weight", type=float, default=-1.0, help="原始 foreground logit margin loss 权重")
    parser.add_argument("--positive-logit-margin", type=float, default=-999.0, help="正样本 logit 目标 margin")
    parser.add_argument("--negative-margin-weight", type=float, default=-1.0, help="hard negative margin loss 内部权重")
    parser.add_argument("--hard-negative-fraction", type=float, default=-1.0, help="参与 margin loss 的最高分负样本比例")
    parser.add_argument("--confidence-ranking-weight", type=float, default=-1.0, help="event-level 正负 logit ranking loss 权重")
    parser.add_argument("--ranking-margin", type=float, default=-1.0, help="event-level logit ranking margin")
    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = args.distributed or world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # NCCL 在 process group 初始化之前就需要知道当前进程绑定的 GPU。
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if distributed and world_size > 1 and not dist.is_initialized():
        # 本项目所用的 PyTorch 1.9/NCCL 2.7 在双 3090 的 PCIe 拓扑上
        # 默认 P2P/共享内存路径不稳定；socket collective 仍是 NCCL，
        # 但可避免 invalid usage/hang。用户可提前设置同名变量覆盖它。
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank() if dist.is_initialized() else 0
    return distributed and world_size > 1, rank, local_rank, world_size


def move_batch(batch, device):
    # names 不需要搬到 GPU；其余张量按需搬运。
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_sparse_input(batch, device, spatial_shape):
    coords = batch["voxel_coords"].to(device, non_blocking=True).int()
    features = batch["voxel_features"].to(device, non_blocking=True).float()
    batch_size = int(batch["batch_size"].item())
    return spconv.SparseConvTensor(features, coords, spatial_shape, batch_size)


def event_probabilities(output, batch):
    voxel_probability = output["voxel_probability"].squeeze(1)
    mapping = batch["event_to_voxel"].to(voxel_probability.device, non_blocking=True)
    return voxel_probability[mapping]


def update_metrics(metrics, probabilities, batch):
    probabilities = probabilities.detach().float().cpu()
    labels = batch["labels"].detach().float().cpu()
    coords = batch["event_coords"].detach().cpu()
    instance_ids = batch["instance_ids"].detach().cpu()
    for batch_index in range(int(batch["batch_size"].item())):
        keep = coords[:, 0] == batch_index
        metrics.update(probabilities[keep], labels[keep], coords[keep], instance_ids[keep])


@torch.no_grad()
def validate(model, loader, device, config, rank=0):
    model.eval()
    metrics = EVUAVMetrics(
        resolution=config["data"]["resolution"],
        window_ms=config["data"]["window_ms"],
        slice_ms=config["evaluation"]["slice_ms"],
        threshold=config["evaluation"]["threshold"],
        coverage_threshold=config["evaluation"]["coverage_threshold"],
    )
    for batch in loader:
        gpu_batch = move_batch(batch, device)
        sparse_input = make_sparse_input(gpu_batch, device, model.module.input_spatial_shape if isinstance(model, DDP) else model.input_spatial_shape)
        output = model(sparse_input)
        update_metrics(metrics, event_probabilities(output, gpu_batch), batch)
    result = metrics.compute()
    model.train()
    return result


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, metrics, config):
    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.pgtd_output_weight >= 0:
        config["model"]["pgtd_output_weight"] = args.pgtd_output_weight
    if args.pgtd_logit_weight >= 0:
        config["model"]["pgtd_logit_weight"] = args.pgtd_logit_weight
    if args.pgtd_probability_weight >= 0:
        config["loss"]["pgtd_probability_weight"] = args.pgtd_probability_weight
    if args.confidence_margin_weight >= 0:
        config["loss"]["confidence_margin_weight"] = args.confidence_margin_weight
    if args.positive_logit_margin > -999.0:
        config["loss"]["positive_logit_margin"] = args.positive_logit_margin
    if args.negative_margin_weight >= 0:
        config["loss"]["negative_margin_weight"] = args.negative_margin_weight
    if args.hard_negative_fraction >= 0:
        config["loss"]["hard_negative_fraction"] = args.hard_negative_fraction
    if args.confidence_ranking_weight >= 0:
        config["loss"]["confidence_ranking_weight"] = args.confidence_ranking_weight
    if args.ranking_margin >= 0:
        config["loss"]["ranking_margin"] = args.ranking_margin
    if args.learning_rate > 0:
        config["train"]["learning_rate"] = args.learning_rate
    if args.motion_weight >= 0:
        config["loss"]["motion_weight"] = args.motion_weight
    if args.dice_weight >= 0:
        config["loss"]["dice_weight"] = args.dice_weight
    if args.focal_alpha >= 0:
        config["loss"]["focal_alpha"] = args.focal_alpha
    if args.augment:
        config["data"]["augmentation"] = True
    if args.epochs > 0:
        config["train"]["epochs"] = args.epochs
    distributed, rank, local_rank, world_size = init_distributed(args)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    setup_seed(int(config.get("seed", 37)) + rank)
    is_main = rank == 0

    train_cfg = config["train"]
    max_events = int(args.max_events or config["data"]["max_events"])
    workers = int(args.workers if args.workers >= 0 else train_cfg.get("workers", 4))
    output_dir = os.path.abspath(args.output_dir or train_cfg.get("output_dir", "outputs/tha_ev_uav"))
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config_used.yaml"), "w", encoding="utf-8") as handle:
            import yaml
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    data_cfg = config["data"]
    train_set = EVUAVDataset(data_cfg["root"], "train", max_events, config.get("seed", 37), data_cfg.get("augmentation", False), data_cfg.get("resolution", [346, 260]))
    val_set = EVUAVDataset(data_cfg["root"], "val", max_events, config.get("seed", 37), False, data_cfg.get("resolution", [346, 260]))
    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_ev_uav,
        drop_last=False,
    )
    # 只由 rank 0 完整评测，其他 rank 在 barrier 等待，避免重复读验证集。
    val_loader = None
    if is_main:
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            collate_fn=collate_ev_uav,
        )

    model = THANet(config).to(device)
    if args.init_checkpoint:
        init_state = torch.load(args.init_checkpoint, map_location=device)
        model_state = init_state.get("model", init_state)
        model.load_state_dict(model_state, strict=True)
        if is_main:
            print("加载初始模型权重:", args.init_checkpoint, flush=True)
    if args.freeze_backbone:
        trainable_prefixes = ("head",) if args.head_only else (("tha", "pacm", "pgtd") + (("head",) if args.train_confidence_head else ()))
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.split(".", 1)[0] in trainable_prefixes
        if is_main:
            trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            print("冻结主干，仅训练 THA/PACM/PGTD；可训练参数:", trainable, flush=True)
    if distributed:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    total_epochs = int(args.epochs or train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    best_iou = -1.0
    resume_path = args.resume or os.path.join(output_dir, "latest.pth")
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_iou = float(checkpoint.get("metrics", {}).get("IoU", -1.0))
        if is_main:
            print("恢复 checkpoint:", resume_path, "从 epoch", start_epoch, flush=True)

    if is_main:
        print("训练样本={}, 验证样本={}, world_size={}, AMP={}".format(len(train_set), len(val_set), world_size, amp_enabled), flush=True)
    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        if args.freeze_backbone:
            # BatchNorm 的 running statistics 仍会在 train() 中更新；冻结主干
            # 时只显式固定 BN，稀疏卷积本身保持 train()，避免 spconv 2.3
            # 在 train/eval 混合模式下重新选择到空算法。
            model_to_freeze = model.module if isinstance(model, DDP) else model
            trainable_prefixes = ("tha", "pacm", "pgtd")
            for name, module in model_to_freeze.named_children():
                if name not in trainable_prefixes:
                    for submodule in module.modules():
                        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
                            submodule.eval()
        start_time = time.time()
        running = {"loss": 0.0, "seg": 0.0, "motion": 0.0, "persistence": 0.0, "pgtd": 0.0, "confidence": 0.0, "ranking": 0.0}
        for step, batch in enumerate(train_loader):
            gpu_batch = move_batch(batch, device)
            sparse_input = make_sparse_input(gpu_batch, device, model.module.input_spatial_shape if isinstance(model, DDP) else model.input_spatial_shape)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(sparse_input)
                loss, parts = thanet_loss(output, gpu_batch, config)
            if not torch.isfinite(loss):
                if is_main:
                    print("非有限 loss 明细:", parts, "输出有限性:", {
                        key: bool(torch.isfinite(value).all().item())
                        for key, value in output.items() if torch.is_tensor(value)
                    }, flush=True)
                raise FloatingPointError("非有限 loss: {}".format(float(loss.detach().cpu())))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 10.0)))
            scaler.step(optimizer)
            scaler.update()
            for key in running:
                running[key] += parts[key]
            if is_main and step == 0:
                print("epoch {:03d} step {:03d}: loss={:.5f} seg={:.5f} motion={:.5f} pg={:.5f} ptd={:.5f} conf={:.5f} rank={:.5f}".format(
                    epoch, step, parts["loss"], parts["seg"], parts["motion"], parts["persistence"], parts["pgtd"], parts["confidence"], parts["ranking"]
                ), flush=True)
            if args.max_steps > 0 and (step + 1) >= args.max_steps:
                break
        scheduler.step()
        if is_main:
            denom = max(len(train_loader), 1)
            averages = {key: value / denom for key, value in running.items()}
            # 验证只在 rank 0 进行，用裸 module 避免 DDP wrapper 在单进程
            # validation forward 中尝试跨 rank 同步。
            metrics = validate(model.module if isinstance(model, DDP) else model, val_loader, device, config)
            elapsed = time.time() - start_time
            log_record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "time_sec": elapsed}
            log_record.update(averages)
            log_record.update(metrics)
            with open(os.path.join(output_dir, "train.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            print("epoch {:03d}: {} ({}s)".format(epoch, json.dumps(log_record, ensure_ascii=False), int(elapsed)), flush=True)
            save_checkpoint(os.path.join(output_dir, "latest.pth"), model, optimizer, scheduler, scaler, epoch, metrics, config)
            if metrics["IoU"] > best_iou:
                best_iou = metrics["IoU"]
                save_checkpoint(os.path.join(output_dir, "best.pth"), model, optimizer, scheduler, scaler, epoch, metrics, config)
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
