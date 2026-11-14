# -*- coding: utf-8 -*-
"""YOLO 資料集的小工具（不需要 torch）：找 yaml、找資料夾、讀標註檔。"""

import hashlib
import os

IMAGE_END = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_NAMES = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}


def find_yaml(root):
    """在 root 底下找含有 names 的 yaml 檔（通常是 data.yaml）。回傳 (路徑, 內容)。"""
    import yaml
    candidates = []
    for folder, dirs, files in os.walk(root):
        for name in sorted(files):
            if name.endswith(".yaml") or name.endswith(".yml"):
                candidates.append(os.path.join(folder, name))
    candidates.sort(key=lambda p: (os.path.basename(p) != "data.yaml", p))
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            continue
        if isinstance(data, dict) and "names" in data:
            return path, data
    raise SystemExit("在 " + str(root) + " 底下找不到含 names 的 yaml 檔（例如 data.yaml），請確認解壓縮的位置")


def load_names(data):
    """把 yaml 的 names 轉成類別名稱清單。"""
    names = data["names"]
    if isinstance(names, dict):
        keys = sorted(names.keys(), key=lambda k: int(k))
        return [names[k] for k in keys]
    return list(names)


def has_images(folder):
    if not os.path.isdir(folder):
        return False
    for name in os.listdir(folder):
        if name.lower().endswith(IMAGE_END):
            return True
    return False


def find_split_dirs(root, split):
    """找某個資料組（train / val / test）的照片資料夾與標註資料夾。找不到就回傳 (None, None)。
    支援兩種排法：root/images/train + root/labels/train，或 root/train/images + root/train/labels。"""
    for alias in SPLIT_NAMES[split]:
        options = [
            (os.path.join(root, "images", alias), os.path.join(root, "labels", alias)),
            (os.path.join(root, alias, "images"), os.path.join(root, alias, "labels")),
            (os.path.join(root, alias), os.path.join(root, alias)),
        ]
        for image_dir, label_dir in options:
            if has_images(image_dir):
                return image_dir, label_dir
    return None, None


def list_images(folder):
    """資料夾裡所有照片（排序後）。"""
    names = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_END):
            names.append(name)
    return names


def read_labels(label_path, num_classes):
    """讀一個標註檔。回傳 (框清單, 問題清單)。框 = (類別, 中心x, 中心y, 寬, 高)。"""
    boxes = []
    problems = []
    if not os.path.exists(label_path):
        return boxes, ["missing"]
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        if len(parts) < 5:
            problems.append("欄位不足")
            continue
        try:
            class_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            problems.append("不是數字")
            continue
        if class_id < 0 or class_id >= num_classes:
            problems.append("類別編號超出範圍")
            continue
        if cx < 0 or cx > 1 or cy < 0 or cy > 1 or w <= 0 or w > 1 or h <= 0 or h > 1:
            problems.append("座標超出範圍")
            continue
        boxes.append((class_id, cx, cy, w, h))
    return boxes, problems


def quick_signature(path):
    """快速指紋（檔案大小加頭尾各 64KB 的 md5），用來找完全重複的照片。"""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(65536)
        f.seek(max(0, size - 65536))
        tail = f.read(65536)
    return size, hashlib.md5(head + tail).hexdigest()


def counts_by_class(scores, classes, conf, num_classes):
    """一張照片的所有偵測結果，在信心門檻 conf 下，各類別各有幾個。"""
    counts = [0] * num_classes
    for i in range(len(scores)):
        if scores[i] >= conf:
            counts[int(classes[i])] = counts[int(classes[i])] + 1
    return counts
