"""RF-DETR (Roboflow) COCO-pretrained detector.

License: the ``rfdetr`` package is Apache-2.0. The COCO checkpoints for the
Nano/Small/Medium/Large variants that it downloads on first use are published
by Roboflow under Apache-2.0 as well. The XL/2XL variants live in a separate
"plus" package under a different licence (PML-1.0) and are NOT used here.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from replay.detection.base import BALL, PERSON, Detections, Detector

COCO_PERSON = 1
COCO_SPORTS_BALL = 37

_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


class RFDETRDetector(Detector):
    license_note = "rfdetr code + COCO weights: Apache-2.0 (Roboflow)"

    def __init__(self, variant: str = "medium", threshold: float = 0.3, resolution: int | None = None):
        import rfdetr
        import torch

        if variant not in _VARIANTS:
            raise ValueError(f"Unknown RF-DETR variant {variant!r}; choose one of {sorted(_VARIANTS)}")
        if not torch.cuda.is_available():
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # physical cores; SMT threads don't help
        kwargs = {"resolution": resolution} if resolution else {}
        self.model = getattr(rfdetr, _VARIANTS[variant])(**kwargs)
        try:
            self.model.optimize_for_inference()
        except Exception:  # optional speed-up; not available on every version/device
            pass
        self.threshold = threshold
        self.name = f"rfdetr-{variant} on {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'} (PyTorch)"

    def detect(self, image_bgr: np.ndarray) -> Detections:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.model.predict(rgb, threshold=self.threshold, include_source_image=False)
        cid = np.asarray(res.class_id)
        keep = (cid == COCO_PERSON) | (cid == COCO_SPORTS_BALL)
        if not keep.any():
            return Detections.empty()
        cls = np.where(cid[keep] == COCO_PERSON, PERSON, BALL).astype(np.int32)
        return Detections(np.asarray(res.xyxy)[keep].astype(np.float32),
                          np.asarray(res.confidence)[keep].astype(np.float32), cls)
