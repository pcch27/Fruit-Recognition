# -*- coding: utf-8 -*-
"""B 的評估小工具（純 Python，不需要 torch 或 ultralytics）：框的重疊、比對、換算。"""


def iou(box_a, box_b):
    """兩個框（x1, y1, x2, y2）的重疊比例（0 到 1）。"""
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def yolo_to_xyxy(cx, cy, w, h, image_w, image_h):
    """YOLO 的相對座標（中心 x、中心 y、寬、高）換成像素座標（x1, y1, x2, y2）。"""
    x1 = (cx - w / 2.0) * image_w
    y1 = (cy - h / 2.0) * image_h
    x2 = (cx + w / 2.0) * image_w
    y2 = (cy + h / 2.0) * image_h
    return (x1, y1, x2, y2)


def family_of_name(name):
    """類別名稱的第一個字（小寫）。例如 'Apple Red' -> 'apple'。"""
    words = name.strip().lower().split()
    if len(words) == 0:
        return ""
    return words[0]


def match_boxes(gt_list, pred_list, threshold):
    """用重疊比例把偵測框配對到標註框（信心高的先配，一個標註框只能被配一次）。
    gt_list：[{'family': 水果, 'box': (x1,y1,x2,y2)}, ...]
    pred_list：[{'name': 類別名稱, 'conf': 信心, 'box': (...)}, ...]
    回傳 (配對成功數, 其中種類也對的數量, 沒配到的偵測框數, 沒被找到的標註框數)。"""
    order = sorted(range(len(pred_list)), key=lambda i: -pred_list[i]["conf"])
    used = [False] * len(gt_list)
    matched = 0
    class_correct = 0
    for i in order:
        best_j = -1
        best_iou = 0.0
        for j in range(len(gt_list)):
            if used[j]:
                continue
            value = iou(pred_list[i]["box"], gt_list[j]["box"])
            if value >= threshold and value > best_iou:
                best_iou = value
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            matched = matched + 1
            if family_of_name(pred_list[i]["name"]) == gt_list[best_j]["family"]:
                class_correct = class_correct + 1
    false_positive = len(pred_list) - matched
    missed = len(gt_list) - matched
    return matched, class_correct, false_positive, missed
