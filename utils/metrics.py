"""EV-UAV 的事件级 IoU/ACC 与 50 ms Pd/Fa 评测。"""

import math

import numpy as np
from scipy import ndimage


class EVUAVMetrics:
    def __init__(self, resolution=(346, 260), window_ms=8000, slice_ms=50, threshold=0.9, coverage_threshold=1e-4):
        self.width = int(resolution[0])
        self.height = int(resolution[1])
        self.window_ms = int(window_ms)
        self.slice_ms = int(slice_ms)
        self.threshold = float(threshold)
        self.coverage_threshold = float(coverage_threshold)
        self.samples = []

    def update(self, probabilities, labels, event_coords, instance_ids):
        self.samples.append(
            (
                np.asarray(probabilities, dtype=np.float32).reshape(-1),
                np.asarray(labels, dtype=np.float32).reshape(-1),
                np.asarray(event_coords, dtype=np.int64),
                np.asarray(instance_ids, dtype=np.int64).reshape(-1),
            )
        )

    def compute(self):
        if not self.samples:
            raise RuntimeError("评测器没有收到样本")
        pred_all = np.concatenate([x[0] for x in self.samples])
        label_all = np.concatenate([x[1] for x in self.samples]) > 0.5
        binary = pred_all >= self.threshold
        intersection = np.logical_and(binary, label_all).sum()
        union = np.logical_or(binary, label_all).sum()
        iou = float(intersection / max(union, 1)) * 100.0
        acc = float(np.logical_and(binary, label_all).sum() / max(label_all.sum(), 1)) * 100.0

        target_total = 0
        target_detected = 0
        false_components = 0
        total_slices = 0
        structure = np.ones((3, 3), dtype=np.uint8)
        for probabilities, labels, coords, instance_ids in self.samples:
            pred = probabilities >= self.threshold
            gt = labels > 0.5
            times = coords[:, 3]
            if len(times) == 0:
                continue
            start = int(times.min())
            end = int(times.max())
            n_slices = max(1, int(math.ceil(float(end - start + 1) / self.slice_ms)))
            total_slices += n_slices
            for slice_index in range(n_slices):
                lo = start + slice_index * self.slice_ms
                hi = lo + self.slice_ms
                in_slice = (times >= lo) & (times < hi)
                if not in_slice.any():
                    continue
                for object_id in np.unique(instance_ids[in_slice]):
                    if object_id <= 0:
                        continue
                    object_events = in_slice & (instance_ids == object_id)
                    target_total += 1
                    coverage = float(pred[object_events].sum()) / max(int(object_events.sum()), 1)
                    if coverage >= self.coverage_threshold:
                        target_detected += 1

                false_events = in_slice & (~gt) & pred
                if false_events.any():
                    mask = np.zeros((self.height, self.width), dtype=np.uint8)
                    xy = coords[false_events, 1:3]
                    valid = (
                        (xy[:, 0] >= 0) & (xy[:, 0] < self.width) &
                        (xy[:, 1] >= 0) & (xy[:, 1] < self.height)
                    )
                    mask[xy[valid, 1], xy[valid, 0]] = 1
                    _, components = ndimage.label(mask, structure=structure)
                    false_components += int(components)

        pd = float(target_detected / max(target_total, 1)) * 100.0
        fa = float(false_components / max(total_slices * self.width * self.height, 1)) * 1e4
        return {"IoU": iou, "ACC": acc, "Pd": pd, "Fa": fa}
