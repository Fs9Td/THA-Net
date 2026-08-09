"""在 EV-UAV test split 上评测 THA-Net checkpoint。"""

import argparse
import json

import torch
from torch.utils.data import DataLoader
import spconv.pytorch as spconv

from data import EVUAVDataset, collate_ev_uav
from models import THANet
from utils.config import load_config
from utils.metrics import EVUAVMetrics


def main():
    parser = argparse.ArgumentParser(description="评测 THA-Net on EV-UAV")
    parser.add_argument("--config", default="configs/tha_ev_uav.yaml")
    parser.add_argument("--data-root", default="", help="可选 EV-UAV 根目录，覆盖 config 的 data.root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    saved_model_config = saved_config.get("model", {}) if isinstance(saved_config, dict) else {}
    if saved_model_config:
        config["model"].update(saved_model_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = THANet(config).to(device).eval()
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    data_cfg = config["data"]
    dataset = EVUAVDataset(data_cfg["root"], args.split, data_cfg["max_events"], config.get("seed", 37))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=device.type == "cuda", collate_fn=collate_ev_uav)
    probabilities_cache = []
    labels_cache = []
    coords_cache = []
    ids_cache = []
    with torch.no_grad():
        for batch in loader:
            def infer(event_coords, voxel_features):
                sparse_input = spconv.SparseConvTensor(
                    voxel_features.to(device, non_blocking=True).float(),
                    event_coords.to(device, non_blocking=True).int(),
                    model.input_spatial_shape,
                    int(batch["batch_size"].item()),
                )
                output = model(sparse_input)
                return output["voxel_probability"].squeeze(1)[batch["event_to_voxel"].to(device)].cpu()

            event_prob = infer(batch["voxel_coords"], batch["voxel_features"])
            keep = batch["event_coords"][:, 0] == 0
            probabilities_cache.append(event_prob[keep])
            labels_cache.append(batch["labels"][keep])
            coords_cache.append(batch["event_coords"][keep])
            ids_cache.append(batch["instance_ids"][keep])

    metrics = EVUAVMetrics(
        resolution=data_cfg["resolution"], window_ms=data_cfg["window_ms"],
        slice_ms=config["evaluation"]["slice_ms"], threshold=config["evaluation"]["threshold"],
        coverage_threshold=config["evaluation"]["coverage_threshold"],
    )
    for p, y, c, i in zip(probabilities_cache, labels_cache, coords_cache, ids_cache):
        metrics.update(p, y, c, i)
    result = metrics.compute()
    result["checkpoint"] = args.checkpoint
    result["split"] = args.split
    result["threshold"] = config["evaluation"]["threshold"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
