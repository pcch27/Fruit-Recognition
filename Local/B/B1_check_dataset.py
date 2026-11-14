# -*- coding: utf-8 -*-
"""B1：檢查 YOLO 資料集（例如 DFV16）。只看不改。
會印出：類別清單、各資料組的照片數與框數、標註格式問題、沒有標註的照片、跨資料組重複的照片，
並抽樣把框畫在照片上存成圖（請你用眼睛看：框有沒有貼著水果、類別名稱對不對）。

用法：
  python B/B1_check_dataset.py --root /content/dfv16
"""
import argparse
import colorsys
import json
import os
import random
import sys

from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import yolo_data as yd


def pick_color(i, n):
    r, g, b = colorsys.hsv_to_rgb((i / float(max(1, n))) % 1.0, 0.9, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--sanity", type=int, default=60, help="要畫幾張疊框圖")
    parser.add_argument("--out", default="results/b_sanity")
    args = parser.parse_args()

    yaml_path, data = yd.find_yaml(args.root)
    names = yd.load_names(data)
    num_classes = len(names)
    print("yaml 檔：", yaml_path)
    print("類別數：", num_classes)
    print("類別：", names)
    print("")

    class_count = {}
    signatures = {}
    folders = {}
    report = {"names": names, "splits": {}}
    for split in ["train", "val", "test"]:
        image_dir, label_dir = yd.find_split_dirs(args.root, split)
        if image_dir is None:
            print("[" + split + "] 找不到照片資料夾（略過）")
            continue
        image_names = yd.list_images(image_dir)
        count_box = 0
        count_empty = 0
        count_missing = 0
        problems = {}
        class_count[split] = {}
        for name in image_names:
            stem = os.path.splitext(name)[0]
            boxes, problem_list = yd.read_labels(os.path.join(label_dir, stem + ".txt"), num_classes)
            for p in problem_list:
                if p == "missing":
                    count_missing = count_missing + 1
                else:
                    problems[p] = problems.get(p, 0) + 1
            if len(boxes) == 0:
                count_empty = count_empty + 1
            for box in boxes:
                class_count[split][box[0]] = class_count[split].get(box[0], 0) + 1
            count_box = count_box + len(boxes)
            key = yd.quick_signature(os.path.join(image_dir, name))
            if key not in signatures:
                signatures[key] = []
            signatures[key].append(split + "/" + name)
        folders[split] = (image_dir, label_dir, image_names)
        print("[" + split + "] 照片", len(image_names), "張，框", count_box, "個，沒有框的照片", count_empty,
              "張，缺標註檔", count_missing, "個，格式問題", problems if problems else "無")
        report["splits"][split] = {"images": len(image_names), "boxes": count_box}

    print("")
    print("各類別的框數（train / val / test）：")
    for i in range(num_classes):
        a = class_count.get("train", {}).get(i, 0)
        b = class_count.get("val", {}).get(i, 0)
        c = class_count.get("test", {}).get(i, 0)
        warn = ""
        if a < 100 or c < 20:
            warn = "   <- 偏少，這類結果會不穩定"
        print("  %2d %-18s %6d %6d %6d%s" % (i, names[i], a, b, c, warn))

    duplicated = 0
    for key in signatures:
        places = signatures[key]
        splits_in = set()
        for p in places:
            splits_in.add(p.split("/")[0])
        if len(splits_in) > 1:
            duplicated = duplicated + 1
    print("")
    print("同一張照片同時出現在不同資料組的組數：", duplicated, "（應該是 0）")

    # 抽樣畫框
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(0)
    per_split = max(1, args.sanity // max(1, len(folders)))
    saved = 0
    for split in folders:
        image_dir, label_dir, image_names = folders[split]
        chosen = list(image_names)
        rng.shuffle(chosen)
        for name in chosen[:per_split]:
            image = Image.open(os.path.join(image_dir, name)).convert("RGB")
            image.thumbnail((960, 960))
            width, height = image.size
            draw = ImageDraw.Draw(image)
            boxes, problem_list = yd.read_labels(os.path.join(label_dir, os.path.splitext(name)[0] + ".txt"), num_classes)
            for class_id, cx, cy, w, h in boxes:
                color = pick_color(class_id, num_classes)
                x1 = (cx - w / 2) * width
                y1 = (cy - h / 2) * height
                x2 = (cx + w / 2) * width
                y2 = (cy + h / 2) * height
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1 + 3, y1 + 3), names[class_id], fill=color)
            image.save(os.path.join(args.out, split + "_" + os.path.splitext(name)[0] + ".jpg"), quality=88)
            saved = saved + 1
    print("已存", saved, "張疊框圖到", args.out, "，請逐張看過再開始訓練")
    os.makedirs("results", exist_ok=True)
    with open("results/b_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
