"""Stage 1: YOLO fine-tuning on pressure maps + token extraction (boxes, confidences, pooled features)."""
import json, shutil
from pathlib import Path
import numpy as np


def _device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def train_fold(cfg, fold, W: Path):
    from ultralytics import YOLO
    W = Path(W)
    data_yaml = W / "yolo_data" / f"fold{fold}" / "data.yaml"
    model = YOLO(cfg.yolo.model)
    model.train(
        data=str(data_yaml), epochs=cfg.yolo.epochs, imgsz=cfg.data.canvas, batch=cfg.yolo.batch,
        project=str(W / "yolo_runs"), name=f"fold{fold}", exist_ok=True, device=_device(),
        workers=cfg.yolo.workers, patience=cfg.yolo.patience, seed=cfg.seed, pretrained=True,
        # pressure maps are not natural images: no colour jitter / mosaic; body parts are mirror-symmetric
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.1, scale=0.2, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, mosaic=0.0, mixup=0.0, close_mosaic=0, plots=False, verbose=False)
    best = Path(model.trainer.best)
    dst = W / "yolo_runs" / f"fold{fold}_best.pt"
    shutil.copy(best, dst)
    # detection quality on held-out TEST subjects of this fold
    m = YOLO(str(dst)).val(data=str(data_yaml), split="test", imgsz=cfg.data.canvas, batch=cfg.yolo.batch,
                           plots=False, verbose=False, device=_device())
    names = ["Head", "Lower_Arm", "Lower_Leg", "Torso", "Upper_Arm", "Upper_Leg"]
    per_cls = {}
    try:
        for ci, mp in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index]):
            per_cls[names[int(ci)]] = float(mp)
    except Exception:
        pass
    res = dict(fold=fold, map50=float(m.box.map50), map50_95=float(m.box.map),
               precision=float(m.box.mp), recall=float(m.box.mr), per_class_map50_95=per_cls)
    (W / "yolo_runs" / f"fold{fold}_test_metrics.json").write_text(json.dumps(res, indent=1))
    print("[yolo] test metrics:", res)
    return dst


class TokenExtractor:
    """Runs the fine-tuned detector manually (batched) and returns, per frame:
    detections after NMS and global-average-pooled features of the 3 neck maps feeding the Detect head."""

    def __init__(self, weights, renderer, conf=0.05, iou=0.5, device=None):
        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.R, self.conf, self.iou = renderer, conf, iou
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = YOLO(str(weights)).model.to(self.dev).float().eval()
        self._feats = None
        self.net.model[-1].register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        # copy the list: some ultralytics versions overwrite the list entries in place inside Detect.forward
        self._feats = list(inputs[0])

    @staticmethod
    def _preds(out):
        if isinstance(out, dict):
            out = out.get("one2one", out.get("one2many", out.get("preds")))
        return out[0] if isinstance(out, (list, tuple)) else out

    def _pool(self):
        R, outs = self.R, []
        for f in self._feats:
            s = R.canvas / f.shape[-1]
            c0, c1 = int(np.floor(R.x0 / s)), int(np.ceil((R.x0 + R.w) / s))
            outs.append(f[:, :, :, c0:c1].mean((2, 3)))      # only the part of the canvas that holds the sensor map
        return self.torch.cat(outs, 1)

    def run(self, frames, batch=256):
        try:                                            # ultralytics >= 8.4 moved NMS to utils.nms
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:
            from ultralytics.utils.ops import non_max_suppression
        torch = self.torch
        dets_all, feats_all = [], []
        for i in range(0, len(frames), batch):
            imgs = self.R.batch(frames[i:i + batch])                              # (B,H,W,3) cv2/BGR order
            x = torch.from_numpy(np.ascontiguousarray(imgs[..., ::-1])).permute(0, 3, 1, 2).float().div(255).to(self.dev)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.dev == "cuda")):
                    out = self.net(x)
                preds = self._preds(out).float()
                nms = non_max_suppression(preds, self.conf, self.iou, max_det=30, nc=6)
                feats_all.append(self._pool().float().cpu().numpy().astype(np.float16))
            dets_all += [d.detach().cpu().numpy() for d in nms]
        return dets_all, np.concatenate(feats_all, 0)
