# -*- coding: utf-8 -*-
"""A 專案共用程式（訓練、評估、網頁介面都用這一份，避免前後不一致）。
寫法刻意簡單：不用 f-string，每個函式都有中文說明。"""

import os
import re

# 所有模型的輸入大小（Baseline CNN、ResNet18、MobileNetV2 都用 224）
IMG_SIZE = 224

# 不要的種類：蔬菜、堅果、種子、菇類。只留水果。想改就直接改這個清單。
NOT_FRUIT = [
    "beans", "bean", "beetroot", "cabbage", "carrot", "cauliflower", "celery",
    "corn", "cucumber", "eggplant", "garlic", "ginger", "kohlrabi", "onion",
    "pepper", "peper", "potato", "pumpkin", "radish", "tomato", "zucchini",
    "mushroom", "murshroom", "almonds", "caju", "chestnut", "hazelnut", "nut",
    "pistachio", "walnut", "chia", "peanut", "egg",
]

# 檔名裡只是「形容」的字，不是水果，拆檔名時直接丟掉
DROP_WORDS = ["sour", "wax", "sweet", "fruit", "shell", "lady", "finger", "flat", "pecan", "rainier"]

# 檔名裡的字 -> Fruits-360 的大類名稱（Fruits-360 的類別名稱第一個字就是大類）
NAME_ALIAS = {
    "ananas": "pineapple",
    "currant": "redcurrant",
    "tangerine": "mandarine",
}


def family_from_class_name(class_name):
    """Fruits-360 的類別名稱 -> 大類。例如 'Apple Red 1' -> 'apple'，'apple_braeburn_1' -> 'apple'。
    注意：有些資料夾用空白分開、有些用底線分開（original-size 就有），所以兩種都要切。"""
    words = re.split(r"[ _]+", class_name.strip().lower())
    for word in words:
        if word != "":
            return word
    return ""


def is_fruit(family):
    """這個大類是不是水果（不在 NOT_FRUIT 清單裡就算水果）。"""
    return family not in NOT_FRUIT


def tokens_from_filename(file_name):
    """把 multi 資料夾的檔名拆成水果大類清單。
    例如 'apple_banana_lemon_1.jpg' -> ['apple', 'banana', 'lemon']
         'cherry(wax)_3.jpg'        -> ['cherry']
         'Banana(lady_finger)_2.jpg' -> ['banana']
    注意：這裡只負責拆字，不判斷是不是水果；後面再用大類清單過濾。"""
    stem = os.path.splitext(os.path.basename(file_name))[0].lower()
    stem = re.sub(r"\(.*?\)", "", stem)          # 去掉括號與括號內的字
    stem = re.sub(r"_\d+$", "", stem)             # 去掉最後的編號
    stem = stem.replace(" ", "_")
    result = []
    for word in stem.split("_"):
        if word == "":
            continue
        if word in DROP_WORDS:
            continue
        if word in NAME_ALIAS:
            word = NAME_ALIAS[word]
        if word not in result:
            result.append(word)
    return result


def composition_key(file_name):
    """同一組拍攝（同樣的水果組合）的名字，用來把相近的照片分到同一個資料組。"""
    stem = os.path.splitext(os.path.basename(file_name))[0].lower()
    stem = re.sub(r"\(.*?\)", "", stem)
    stem = re.sub(r"_\d+$", "", stem)
    return stem


def photo_number(file_name):
    """取出檔名最後的編號，例如 'apple_105.JPG' -> 105；沒有編號就回傳 0。"""
    stem = os.path.splitext(os.path.basename(file_name))[0]
    found = re.search(r"_(\d+)$", stem)
    if found is None:
        return 0
    return int(found.group(1))


def split_groups(groups_by_stratum, seed=42):
    """把「資料組」分成 train / val / test（同一組一定在同一邊，避免相近照片洩漏）。
    groups_by_stratum: {層名稱: [組名稱, ...]}。回傳 {組名稱: 'train' 或 'val' 或 'test'}。
    組數少於 3 的層全部放 train（資料太少，無法評估）。"""
    import random
    rng = random.Random(seed)
    result = {}
    for stratum in sorted(groups_by_stratum.keys()):
        groups = sorted(groups_by_stratum[stratum])
        rng.shuffle(groups)
        n = len(groups)
        if n < 3:
            for g in groups:
                result[g] = "train"
            continue
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        for k in range(n):
            if k < n_test:
                result[groups[k]] = "test"
            elif k < n_test + n_val:
                result[groups[k]] = "val"
            else:
                result[groups[k]] = "train"
    return result


# ---------------- 評估用的數字計算（不需要 torch） ----------------

def top_indices(scores, k):
    """分數由高到低，取前 k 個類別編號。scores 是一串數字。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return order[:k]


def top1_hit(scores, true_ids):
    """第一名是不是正確答案之一（單一種類圖就是第一名答對）。"""
    return top_indices(scores, 1)[0] in true_ids


def topk_hit(scores, true_ids, k):
    """前 k 名裡有沒有至少一個正確答案。"""
    for i in top_indices(scores, k):
        if i in true_ids:
            return True
    return False


def found_ratio(scores, true_ids):
    """多種水果的圖：圖裡有 n 種水果，就看前 n 名裡找到幾種，回傳比例（0 到 1）。"""
    n = len(true_ids)
    if n == 0:
        return 0.0
    found = 0
    for i in top_indices(scores, n):
        if i in true_ids:
            found = found + 1
    return found / float(n)


# ---------------- 模型（需要 torch，所以 import 放在函式裡面） ----------------

def build_model(model_name, num_classes, pretrained=True):
    """建立模型。model_name 可以是 'baseline'、'resnet18'、'mobilenetv2'。"""
    import torch.nn as nn
    from torchvision import models

    if model_name == "baseline":
        # 和 Phase 1 相同的 3 層卷積小模型；輸入 224 時，最後特徵圖是 28x28
        features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        small = IMG_SIZE // 8
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * small * small, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        model = nn.Sequential(features, classifier)
        return model

    if model_name == "resnet18":
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenetv2":
        if pretrained:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError("不認得的模型名稱：" + str(model_name))


def freeze_backbone(model, model_name):
    """只留最後的分類層可以訓練（和 Phase 1 的做法相同）。baseline 是從零訓練，不凍結。"""
    if model_name == "baseline":
        return
    for p in model.parameters():
        p.requires_grad = False
    if model_name == "resnet18":
        last = model.fc
    else:
        last = model.classifier[1]
    for p in last.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    """全部參數都可以訓練。"""
    for p in model.parameters():
        p.requires_grad = True


def eval_transform():
    """推論與評估用的前處理：縮成 224x224、轉成 tensor、用 ImageNet 的平均與標準差正規化。"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_transform_white():
    """白底圖的訓練前處理（和 Phase 1 相似：翻轉加旋轉 15 度）。"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_transform_real():
    """真實照片的訓練前處理：隨機裁切、翻轉、調整亮度與顏色。"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
