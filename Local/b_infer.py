# -*- coding: utf-8 -*-
"""B 的推論（網頁介面使用，需要 ultralytics）。載入 YOLO 權重，框出照片裡的水果。"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "models", "b_yolo.pt")
_model = None
_device = "cpu"


def available():
    """有沒有 YOLO 權重檔。"""
    return os.path.exists(WEIGHTS)


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(WEIGHTS)
    return _model


def class_names():
    """權重裡的類別名稱清單（有名稱的模型是水果名稱；舊模型只有 '0'~'15'）。"""
    names = _get_model().names
    return [names[k] for k in sorted(names.keys())] if isinstance(names, dict) else list(names)


def use_device():
    """有 GPU 就用 GPU（要在 ZeroGPU 的 @spaces.GPU 函式「裡面」呼叫）；沒有就用 CPU。"""
    global _device
    import torch
    _device = 0 if torch.cuda.is_available() else "cpu"
    model = _get_model()
    model.predictor = None      # 換裝置後讓 ultralytics 重新建立預測器
    return _device


def release():
    """用完放回 CPU，並丟掉預測器（ZeroGPU 的 GPU 函式結束後 cuda 會消失）。"""
    global _device
    _device = "cpu"
    if _model is not None:
        try:
            _model.to("cpu")
        except Exception:
            pass
        _model.predictor = None


def detect(image, conf=0.25, iou=0.5, imgsz=640, agnostic=True):
    """回傳 [{'name': 類別, 'conf': 信心, 'box': (x1, y1, x2, y2)}, ...]（座標是像素）。
    conf：信心門檻；iou：重疊框合併門檻（越小越容易把重疊框合併掉）；
    imgsz：推論時的圖片大小（越大小水果越容易找到，但越慢）；
    agnostic：True 時，不同類別但位置重疊的框也會合併，只留最有把握的一個。"""
    model = _get_model()
    result = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz,
                           agnostic_nms=agnostic, verbose=False, device=_device)[0]
    found = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.tolist()
        confs = result.boxes.conf.tolist()
        classes = result.boxes.cls.tolist()
        for i in range(len(boxes)):
            found.append({"name": result.names[int(classes[i])], "conf": float(confs[i]), "box": tuple(boxes[i])})
    return found


# ---------- 分塊辨識（大張、水果很多的照片）----------
# 大張照片會先被縮小，水果在偵測圖裡變得太小，整張辨識會漏掉很多顆。
# 做法：整張先辨識一次；如果框得多（>= TILE_MIN_BOXES）而且照片夠大（長邊 >= TILE_MIN_SIDE），
# 再把照片切成 TILE_N x TILE_N 塊（邊緣重疊 20%）各自辨識，和整張的結果合併。
# 水果少、或小張的照片不分塊，結果和原本完全一樣。
TILE_MIN_BOXES = 8
TILE_MIN_SIDE = 1000
TILE_N = 2
TILE_OVERLAP = 0.2
MERGE_IOU = 0.5     # 和已留下的框重疊超過這個比例（IoU）就視為重複
MERGE_IOS = 0.8     # 分塊找到的框，有 80% 以上的面積落在已留下的框裡，也視為重複


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def tile_windows(width, height, n=TILE_N, overlap=TILE_OVERLAP):
    """回傳 n x n 個切塊的位置 (x0, y0, x1, y1)，相鄰兩塊重疊 overlap。"""
    tile_w = width / (1 + (n - 1) * (1 - overlap))
    tile_h = height / (1 + (n - 1) * (1 - overlap))
    windows = []
    for row in range(n):
        for col in range(n):
            x0 = col * tile_w * (1 - overlap)
            y0 = row * tile_h * (1 - overlap)
            windows.append((int(x0), int(y0), int(min(width, x0 + tile_w)), int(min(height, y0 + tile_h))))
    return windows


def merge_tile_boxes(base, extra, agnostic=True, iou_limit=MERGE_IOU, ios_limit=MERGE_IOS):
    """base（整張辨識的框）全部保留；extra（分塊找到的框）依信心由高到低加入，
    和已留下的框太像（IoU 太高，或大部分面積落在別的框裡）就丟掉。"""
    kept = list(base)
    for item in sorted(extra, key=lambda x: -x["conf"]):
        duplicate = False
        for other in kept:
            if (not agnostic) and other["name"] != item["name"]:
                continue
            inter = _inter(item["box"], other["box"])
            if inter <= 0:
                continue
            union = _area(item["box"]) + _area(other["box"]) - inter
            iou = inter / union if union > 0 else 0.0
            own = _area(item["box"])
            ios = inter / own if own > 0 else 0.0
            if iou > iou_limit or ios >= ios_limit:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def detect_tiled(image, conf=0.25, iou=0.5, imgsz=640, agnostic=True, first=None, n=TILE_N,
                 overlap=TILE_OVERLAP, drop_cut=True, ios_limit=MERGE_IOS):
    """整張＋n x n 分塊。first：整張辨識的結果（已經跑過就傳進來，不重跑）。
    分塊裡貼著「切開處」的框是被切到一半的水果，直接丟掉（完整的那顆會在相鄰的塊或整張裡找到）。
    overlap、drop_cut、ios_limit 是給診斷工具比較用的（drop_cut=False 不丟切邊的框；ios_limit 設很大＝不用面積比去重複）。"""
    base = list(first) if first is not None else detect(image, conf, iou, imgsz, agnostic)
    width, height = image.size
    extra = []
    for (x0, y0, x1, y1) in tile_windows(width, height, n, overlap):
        part = image.crop((x0, y0, x1, y1))
        part_w, part_h = part.size
        margin = max(4, int(0.015 * min(part_w, part_h)))
        for item in detect(part, conf, iou, imgsz, agnostic):
            b = item["box"]
            cut = ((x0 > 0 and b[0] <= margin) or (y0 > 0 and b[1] <= margin) or
                   (x1 < width and b[2] >= part_w - margin) or (y1 < height and b[3] >= part_h - margin))
            if cut and drop_cut:
                continue
            extra.append({"name": item["name"], "conf": item["conf"], "box": (b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0)})
    return merge_tile_boxes(base, extra, agnostic, ios_limit=ios_limit)


def detect_auto(image, conf=0.25, iou=0.5, imgsz=640, agnostic=True,
                min_boxes=TILE_MIN_BOXES, min_side=TILE_MIN_SIDE):
    """回傳 (框的清單, 有沒有分塊)。先整張辨識；框得多而且照片夠大才分塊。"""
    found = detect(image, conf, iou, imgsz, agnostic)
    if len(found) >= min_boxes and max(image.size) >= min_side:
        return detect_tiled(image, conf, iou, imgsz, agnostic, first=found), True
    return found, False
