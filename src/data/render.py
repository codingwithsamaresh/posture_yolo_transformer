"""Turn raw 64x27 pressure frames into the 3-channel YOLO input.

Channels (stored in this order in the uint8 image array, cv2/BGR convention):
  0 raw      x / vmax
  1 log      log1p(x) / log1p(vmax)    (lifts light-contact regions: arms, heels)
  2 gradient |Sobel| of the upsampled raw map / grad_norm   (edges of contact blobs)
The upsampled map (scale x) is centred on a canvas x canvas black square so YOLO sees an explicit
square input (no hidden letterboxing => training and manual inference are identical).
"""
import numpy as np
import cv2

ROWS, COLS = 64, 27


class Renderer:
    def __init__(self, vmax=100.0, grad_norm=0.65, scale=4, canvas=256):
        self.vmax, self.grad_norm, self.scale, self.canvas = float(vmax), float(grad_norm), int(scale), int(canvas)
        self.h, self.w = ROWS * self.scale, COLS * self.scale          # 256 x 108
        assert self.h <= self.canvas and self.w <= self.canvas
        self.y0 = (self.canvas - self.h) // 2                             # 0
        self.x0 = (self.canvas - self.w) // 2                             # 74

    @classmethod
    def from_cfg(cls, cfg):
        d = cfg.data
        return cls(d.vmax, d.grad_norm, d.scale, d.canvas)

    # ---- images
    def one(self, frame):
        x = np.clip(frame.astype(np.float32), 0, None)
        raw = np.clip(x / self.vmax, 0, 1)
        log = np.clip(np.log1p(x) / np.log1p(self.vmax), 0, 1)
        up = lambda a: cv2.resize(a, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        raw_u, log_u = np.clip(up(raw), 0, 1), np.clip(up(log), 0, 1)
        gx = cv2.Sobel(raw_u, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(raw_u, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.clip(np.sqrt(gx * gx + gy * gy) / self.grad_norm, 0, 1)
        img = np.zeros((self.canvas, self.canvas, 3), np.uint8)
        sl = (slice(self.y0, self.y0 + self.h), slice(self.x0, self.x0 + self.w))
        img[sl + (0,)] = (raw_u * 255 + 0.5).astype(np.uint8)
        img[sl + (1,)] = (log_u * 255 + 0.5).astype(np.uint8)
        img[sl + (2,)] = (grad * 255 + 0.5).astype(np.uint8)
        return img

    def batch(self, frames):
        return np.stack([self.one(f) for f in frames])

    # ---- coordinate transforms between sensor grid (x,y in cells) and canvas
    def grid_box_to_yolo(self, x, y, w, h):
        """COCO bbox (x,y,w,h) in grid cells -> normalised (cx,cy,w,h) on the canvas."""
        x1, y1 = np.clip(x, 0, COLS), np.clip(y, 0, ROWS)
        x2, y2 = np.clip(x + w, 0, COLS), np.clip(y + h, 0, ROWS)
        px1, px2 = self.x0 + x1 * self.scale, self.x0 + x2 * self.scale
        py1, py2 = self.y0 + y1 * self.scale, self.y0 + y2 * self.scale
        c = float(self.canvas)
        return ((px1 + px2) / 2 / c, (py1 + py2) / 2 / c, (px2 - px1) / c, (py2 - py1) / c)

    def canvas_xyxy_to_grid_norm(self, xyxy):
        """(n,4) canvas pixels -> (n,4) normalised sensor coords (cx,cy,w,h) in [0,1]."""
        xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        cx = ((xyxy[:, 0] + xyxy[:, 2]) / 2 - self.x0) / self.w
        cy = ((xyxy[:, 1] + xyxy[:, 3]) / 2 - self.y0) / self.h
        bw = (xyxy[:, 2] - xyxy[:, 0]) / self.w
        bh = (xyxy[:, 3] - xyxy[:, 1]) / self.h
        return np.clip(np.stack([cx, cy, bw, bh], 1), 0, 1)
