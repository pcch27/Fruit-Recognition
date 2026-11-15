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
