# -*- coding: utf-8 -*-
"""A 專案的資料讀取與評估工具（需要 torch）。訓練與評估兩個程式共用。"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import a_common as ac


class PhotoSet(Dataset):
    """一包照片。items 的每一筆是 [照片路徑, [類別編號, ...], 'white' 或 'real']。
    白底圖與真實照片可以用不同的前處理（白底用旋轉，真實照片用隨機裁切）。"""

    def __init__(self, items, num_classes, white_transform, real_transform):
        self.items = items
        self.num_classes = num_classes
        self.white_transform = white_transform
        self.real_transform = real_transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path = self.items[index][0]
        ids = self.items[index][1]
        kind = self.items[index][2]
        image = Image.open(path).convert("RGB")
        if kind == "real":
            x = self.real_transform(image)
        else:
            x = self.white_transform(image)
        target = torch.zeros(self.num_classes)
        for k in ids:
            target[k] = 1.0
        return x, target


def group_by_class(white_items):
    """白底清單 [[路徑, 類別編號], ...] -> {類別編號: [路徑, ...]}。"""
    result = {}
    for path, class_id in white_items:
        if class_id not in result:
            result[class_id] = []
        result[class_id].append(path)
    return result


def pick_white(by_class, max_per_class, rng):
    """每個類別最多隨機挑 max_per_class 張（每個 epoch 重新挑，所以看過的圖會不一樣）。"""
    items = []
    for class_id in sorted(by_class.keys()):
        paths = list(by_class[class_id])
        rng.shuffle(paths)
        for path in paths[:max_per_class]:
            items.append([path, [class_id], "white"])
    return items


def pick_white_n(by_class, how_many, rng):
    """從全部白底圖裡隨機挑 how_many 張（第二階段拿來混進真實照片，避免忘記白底）。"""
    all_items = []
    for class_id in sorted(by_class.keys()):
        for path in by_class[class_id]:
            all_items.append([path, [class_id], "white"])
    rng.shuffle(all_items)
    return all_items[:how_many]


def train_one_epoch(model, items, num_classes, optimizer, criterion, device, batch_size, workers, fix_batchnorm=False):
    """訓練一個 epoch，回傳平均 loss。fix_batchnorm=True 時，BatchNorm 層固定不更新（主幹凍結時要用）。"""
    dataset = PhotoSet(items, num_classes, ac.train_transform_white(), ac.train_transform_real())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    model.train()
    if fix_batchnorm:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    total_loss = 0.0
    total_count = 0
    for x, target in loader:
        x = x.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item() * x.size(0)
        total_count = total_count + x.size(0)
    return total_loss / max(1, total_count)


def predict_scores(model, items, num_classes, device, batch_size, workers):
    """對一批照片算每個類別的分數（0 到 1）。items 每一筆是 [路徑, [編號, ...]]。回傳 numpy 陣列 (張數, 類別數)。"""
    plain = []
    for item in items:
        plain.append([item[0], item[1], "white"])
    transform = ac.eval_transform()
    dataset = PhotoSet(plain, num_classes, transform, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model.eval()
    all_scores = []
    with torch.no_grad():
        for x, target in loader:
            x = x.to(device)
            scores = torch.sigmoid(model(x))
            all_scores.append(scores.cpu().numpy())
    if len(all_scores) == 0:
        return np.zeros((0, num_classes))
    return np.concatenate(all_scores, axis=0)


def evaluate_white(model, items, num_classes, device, batch_size, workers, max_n, seed):
    """白底測試集：看第一名是不是正確的大類。張數太多時隨機抽 max_n 張（固定種子）。"""
    items = list(items)
    if len(items) == 0:
        return {"n": 0}
    if len(items) > max_n:
        random.Random(seed).shuffle(items)
        items = items[:max_n]
    pairs = []
    for path, class_id in items:
        pairs.append([path, [class_id]])
    scores = predict_scores(model, pairs, num_classes, device, batch_size, workers)
    hit1 = 0
    hit3 = 0
    for i in range(len(pairs)):
        row = scores[i].tolist()
        if ac.top1_hit(row, pairs[i][1]):
            hit1 = hit1 + 1
        if ac.topk_hit(row, pairs[i][1], 3):
            hit3 = hit3 + 1
    n = len(pairs)
    return {"n": n, "top1": hit1 / float(n), "top3": hit3 / float(n)}


def evaluate_real(model, items, num_classes, device, batch_size, workers):
    """真實照片：單一種類的圖看第一名與前三名；多種水果的圖看「找到幾成」。
    items 每一筆是 [路徑, [類別編號, ...]]。"""
    if len(items) == 0:
        return {"single_n": 0, "multi_n": 0}
    scores = predict_scores(model, items, num_classes, device, batch_size, workers)
    single_n = 0
    single_hit1 = 0
    single_hit3 = 0
    multi_n = 0
    multi_found = 0.0
    multi_any3 = 0
    for i in range(len(items)):
        row = scores[i].tolist()
        true_ids = items[i][1]
        if len(true_ids) == 1:
            single_n = single_n + 1
            if ac.top1_hit(row, true_ids):
                single_hit1 = single_hit1 + 1
            if ac.topk_hit(row, true_ids, 3):
                single_hit3 = single_hit3 + 1
        else:
            multi_n = multi_n + 1
            multi_found = multi_found + ac.found_ratio(row, true_ids)
            if ac.topk_hit(row, true_ids, 3):
                multi_any3 = multi_any3 + 1
    result = {"single_n": single_n, "multi_n": multi_n}
    if single_n > 0:
        result["single_top1"] = single_hit1 / float(single_n)
        result["single_top3"] = single_hit3 / float(single_n)
    if multi_n > 0:
        result["multi_found_ratio"] = multi_found / float(multi_n)
        result["multi_any_in_top3"] = multi_any3 / float(multi_n)
    return result


def val_score(result):
    """第二階段挑「最好的 epoch」用的分數：單一種類第一名正確率與多種水果找回率的平均。"""
    values = []
    if result.get("single_n", 0) > 0:
        values.append(result["single_top1"])
    if result.get("multi_n", 0) > 0:
        values.append(result["multi_found_ratio"])
    if len(values) == 0:
        return 0.0
    return sum(values) / float(len(values))
