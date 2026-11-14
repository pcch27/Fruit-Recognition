# -*- coding: utf-8 -*-
"""A 的推論（網頁介面使用，需要 torch）。把 models 資料夾裡的三個模型載入，對一張照片算每個水果大類的分數。"""

import os

import torch

import a_common as ac

# （畫面上的名稱, 權重檔名, 模型種類）
MODEL_FILES = [
    ("MobileNetV2", "a_mobilenetv2.pth", "mobilenetv2"),
    ("ResNet18", "a_resnet18.pth", "resnet18"),
    ("Baseline CNN", "a_baseline.pth", "baseline"),
]

LOADED = {}   # 畫面名稱 -> (模型, 大類清單)
_device = "cpu"   # 平常在 CPU；ZeroGPU 只有在 @spaces.GPU 函式執行期間才有 cuda，所以由 use_device() 臨時切換


def use_device():
    """有 GPU 就把三個模型搬過去（要在 GPU 函式「裡面」呼叫）；沒有就留在 CPU。回傳裝置名稱。"""
    global _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in LOADED:
        LOADED[name][0].to(_device)
    return _device


def release():
    """用完把模型放回 CPU（ZeroGPU 的 GPU 函式結束後 cuda 會消失）。"""
    global _device
    _device = "cpu"
    for name in LOADED:
        LOADED[name][0].to("cpu")


def load_models(models_dir):
    """載入 models_dir 裡找得到的權重檔，找不到的就略過。"""
    for display_name, file_name, kind in MODEL_FILES:
        path = os.path.join(models_dir, file_name)
        if not os.path.exists(path):
            print("找不到", file_name, "，略過", display_name)
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu")
            families = checkpoint["families"]
            model = ac.build_model(kind, len(families), pretrained=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            LOADED[display_name] = (model, families)
            print("已載入", display_name, "，大類數", len(families))
        except Exception as error:
            print("載入", display_name, "失敗：", error)


def available_models():
    """目前載入成功的模型名稱（依 MODEL_FILES 的順序）。"""
    names = []
    for display_name, file_name, kind in MODEL_FILES:
        if display_name in LOADED:
            names.append(display_name)
    return names


def predict(image, top_k=5):
    """對一張 PIL 照片，回傳 {模型名稱: [(大類, 分數), ...前 top_k 名]}。"""
    transform = ac.eval_transform()
    x = transform(image.convert("RGB")).unsqueeze(0).to(_device)
    result = {}
    for name in available_models():
        model, families = LOADED[name]
        with torch.no_grad():
            scores = torch.sigmoid(model(x))[0].cpu().tolist()
        order = ac.top_indices(scores, top_k)
        rows = []
        for i in order:
            rows.append((families[i], scores[i]))
        result[name] = rows
    return result
