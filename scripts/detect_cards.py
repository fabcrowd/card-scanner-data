"""Offline onnxruntime-CPU port of the extension's photo card detector.

Ported from auto card lister/tools/extension-wasm:
  lib/detect.js            letterbox 640 (grey 114), NCHW /255, class-0 score else max of other classes,
                           conf floor, cx/cy/w/h -> corners, un-letterbox, clamp, NMS IoU 0.45
  photo-detection.js       filterPhotoDetections (card-shaped ratio 0.55..1.82, area <= 0.65 of photo)
                           dropEdgeFragments (commit e48c421, share 0.7, edge margin max(2, 0.4%))
The Facebook photo path in background.js runs detectCards with the floor lowered to 0.12 (detect.js default is 0.55),
so CONF_PHOTO = 0.12 here. Not ported: tile fallback, hero-photo planning, extendClippedBottom.
Differences: the browser canvas resize is replaced by PIL bilinear.
"""
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# Shipped detector: release v2.1, sha256 5e17ae58...; old local Aug model: auto card lister\tools\vellum-ai\models\card_detect.onnx
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "detector", "card_detect.onnx")
MODEL_SIZE = 640
PAD_COLOR = 114
NMS_IOU = 0.45
CONF_DEFAULT = 0.55  # detect.js CONF_THRESH
CONF_PHOTO = 0.12    # background.js photo path
CARD_RATIO_MIN = 0.55
BOX_RATIO_MAX = 1.82
MAX_PHOTO_AREA_FRACTION = 0.65
EDGE_TOUCH_FRACTION = 0.004
EDGE_FRAGMENT_SHARE = 0.7

Box = Tuple[float, float, float, float]
_sessions = {}


def _session(model_path):
    if model_path not in _sessions:
        import onnxruntime as ort

        _sessions[model_path] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _sessions[model_path]


def _letterbox(img: Image.Image):
    w, h = img.size
    scale = min(MODEL_SIZE / w, MODEL_SIZE / h)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_x, pad_y = (MODEL_SIZE - new_w) // 2, (MODEL_SIZE - new_h) // 2
    canvas = Image.new("RGB", (MODEL_SIZE, MODEL_SIZE), (PAD_COLOR,) * 3)
    canvas.paste(img.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, iw) * max(0.0, ih)
    if inter == 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _nms(dets, threshold):
    dets = sorted(dets, key=lambda d: -d["score"])
    kept = []
    for d in dets:
        if all(_iou(k["box"], d["box"]) <= threshold for k in kept):
            kept.append(d)
    return kept


def detect_raw(img: Image.Image, model_path: str = DEFAULT_MODEL, conf: float = CONF_PHOTO) -> List[dict]:
    """detectCards: [{'box': [x1,y1,x2,y2], 'score'}] in original pixels, after NMS."""
    img = img.convert("RGB")
    ow, oh = img.size
    canvas, scale, pad_x, pad_y = _letterbox(img)
    x = (np.asarray(canvas, dtype=np.float32) / 255).transpose(2, 0, 1)[None]
    sess = _session(model_path)
    out = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0]  # [attrs, anchors]
    if out.shape[0] > out.shape[1]:
        out = out.T
    cx, cy, w, h = out[0], out[1], out[2], out[3]
    cls = out[4:]
    score = cls[0].copy()
    weak = score < conf  # detect.js: only when class 0 is weak, fall back to the best other class
    if cls.shape[0] > 1:
        score = np.where(weak, np.maximum(score, cls[1:].max(axis=0)), score)
    keep = np.nonzero(score >= conf)[0]
    dets = []
    for i in keep:
        x1 = (cx[i] - w[i] / 2 - pad_x) / scale
        y1 = (cy[i] - h[i] / 2 - pad_y) / scale
        x2 = (cx[i] + w[i] / 2 - pad_x) / scale
        y2 = (cy[i] + h[i] / 2 - pad_y) / scale
        dets.append({"box": [max(0.0, float(x1)), max(0.0, float(y1)), min(float(ow), float(x2)), min(float(oh), float(y2))],
                     "score": float(score[i])})
    return _nms(dets, NMS_IOU)


def filter_photo_detections(dets, iw, ih):
    """photo-detection.js filterPhotoDetections."""
    area = iw * ih
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        if CARD_RATIO_MIN <= w / h <= BOX_RATIO_MAX and (w * h) / area <= MAX_PHOTO_AREA_FRACTION:
            out.append(d)
    return out


def _median(vals):
    s = sorted(vals)
    return s[len(s) // 2]  # JS: sorted[floor(n/2)] (upper median)


def drop_edge_fragments(dets, iw, ih):
    """photo-detection.js dropEdgeFragments (e48c421)."""
    mx, my = max(2, iw * EDGE_TOUCH_FRACTION), max(2, ih * EDGE_TOUCH_FRACTION)

    def edges(d):
        x1, y1, x2, y2 = d["box"]
        return (x1 <= mx or x2 >= iw - mx), (y1 <= my or y2 >= ih - my)

    inside = [d for d in dets if not any(edges(d))]
    if len(inside) < 2:
        return list(dets)
    med_w = _median([d["box"][2] - d["box"][0] for d in inside])
    med_h = _median([d["box"][3] - d["box"][1] for d in inside])
    out = []
    for d in dets:
        across, down = edges(d)
        x1, y1, x2, y2 = d["box"]
        if across and x2 - x1 < med_w * EDGE_FRAGMENT_SHARE:
            continue
        if down and y2 - y1 < med_h * EDGE_FRAGMENT_SHARE:
            continue
        out.append(d)
    return out


def detect_cards(img: Image.Image, model_path: str = DEFAULT_MODEL, conf: float = CONF_PHOTO,
                 edge_cut: bool = True) -> List[dict]:
    """Photo path as in background.js: detect, card-shape filter, drop edge fragments. Boxes in original pixels."""
    iw, ih = img.size
    shaped = filter_photo_detections(detect_raw(img, model_path, conf), iw, ih)
    return drop_edge_fragments(shaped, iw, ih) if edge_cut else shaped


TILE_OVERLAP = 0.3
TILE_MIN_SCORE = 0.5


def tile_rects(width, height, n, overlap=TILE_OVERLAP):
    """photo-detection.js tileRects."""
    span = n - (n - 1) * overlap
    tw, th = width / span, height / span
    rects = []
    for row in range(n):
        for col in range(n):
            x1, y1 = round(col * tw * (1 - overlap)), round(row * th * (1 - overlap))
            rects.append((x1, y1, min(width, round(x1 + tw)), min(height, round(y1 + th))))
    return rects


def detect_tiled(img: Image.Image, model_path: str = DEFAULT_MODEL, conf: float = CONF_PHOTO) -> List[dict]:
    """background.js detectTiled + mergeTileDetections (2x2 and 3x3 tiles). In the extension this runs ONLY for
    squarish hero photos; here it is an opt-in to find the small cards of lot photos."""
    img = img.convert("RGB")
    pool = []
    for n in (2, 3):
        for rect in tile_rects(img.width, img.height, n):
            piece = img.crop(rect)
            for d in filter_photo_detections(detect_raw(piece, model_path, conf), piece.width, piece.height):
                if d["score"] < TILE_MIN_SCORE:
                    continue
                b = d["box"]
                pool.append({"box": [b[0] + rect[0], b[1] + rect[1], b[2] + rect[0], b[3] + rect[1]], "score": d["score"]})
    return _nms(pool, NMS_IOU)


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        with Image.open(p) as im:
            print(os.path.basename(p), im.size, [[round(v) for v in d["box"]] for d in detect_cards(im)])
