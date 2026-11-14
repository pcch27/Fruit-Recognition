# -*- coding: utf-8 -*-
"""B0：訓練前先看資料集「原本」有幾類、類別叫什麼，並產生訓練用的 yaml。不會改動、不會覆寫資料集裡原本的 data.yaml。
做的事：
  1. 找到資料集原本的 yaml，印出 nc 與全部類別名稱（編號：名稱），並標出常見水果（香蕉、草莓…）在不在裡面
  2. 統計 train 每個類別有幾張圖、幾個框，並檢查有沒有標註編號超出 yaml 的類別數
  3. 寫出 --out 指定的 yaml（絕對路徑，保留所有原本的類別與名稱），之後訓練就用它
  4. 把資料集資料夾位置寫到 --root_out（B3 等步驟會用到）、類別清單寫到 --names_out（建議存到 Drive）
用法（Colab）：
  python B/B0_prepare_data.py --root /content/dfv16 --out /content/data_fixed.yaml
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import yolo_data as yd

KEYWORDS = ["apple", "banana", "orange", "strawberry", "grape", "lemon", "pear", "peach", "mango", "kiwi",
            "watermelon", "pineapple", "cherry", "tomato", "pomegranate", "avocado", "plum", "melon"]


def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="/content/data_fixed.yaml")
    parser.add_argument("--root_out", default="/content/b_root.txt")
    parser.add_argument("--names_out", default="")
    args = parser.parse_args()

    yaml_path, data = yd.find_yaml(args.root)
    base = os.path.dirname(yaml_path)
    names = yd.load_names(data)
    print("原本的 yaml：", yaml_path)
    print("yaml 寫的類別數 nc =", data.get("nc"), "；names 實際有", len(names), "個")
    print("")
    print("===== 類別編號：名稱 =====")
    for i, name in enumerate(names):
        print("%3d : %s" % (i, name))

    lowered = [str(n).lower() for n in names]
    print("")
    print("===== 常見水果在不在這個資料集 =====")
    for key in KEYWORDS:
        hits = [str(names[i]) for i in range(len(names)) if key in lowered[i]]
        print("%-12s %s" % (key, ("有：" + "、".join(hits)) if hits else "沒有"))

    # 統計 train 標註
    image_dir, label_dir = yd.find_split_dirs(base, "train")
    max_id = len(names) - 1
    boxes_of = {}
    images_of = {}
    if image_dir is not None:
        for name in yd.list_images(image_dir):
            label_path = os.path.join(label_dir, os.path.splitext(name)[0] + ".txt")
            if not os.path.exists(label_path):
                continue
            seen = set()
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f.read().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except ValueError:
                        continue
                    max_id = max(max_id, cid)
                    boxes_of[cid] = boxes_of.get(cid, 0) + 1
                    seen.add(cid)
            for cid in seen:
                images_of[cid] = images_of.get(cid, 0) + 1
    if max_id >= len(names):
        print("")
        print("⚠️ 標註裡出現編號 %d，但 yaml 只有 %d 個名稱；缺的名稱用 class_N 補上。" % (max_id, len(names)))
        names = names + ["class_%d" % i for i in range(len(names), max_id + 1)]

    print("")
    print("===== train 每個類別的圖片數／框數 =====")
    for i, name in enumerate(names):
        print("%3d %-22s 圖 %5d 張   框 %6d 個" % (i, str(name), images_of.get(i, 0), boxes_of.get(i, 0)))

    # 寫訓練用 yaml（保留全部類別）
    new = {"path": base, "names": {i: str(n) for i, n in enumerate(names)}}
    for split, key in [("train", "train"), ("val", "val"), ("test", "test")]:
        img_dir, _ = yd.find_split_dirs(base, split)
        if img_dir is not None:
            new[key] = os.path.relpath(img_dir, base)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(new, f, allow_unicode=True, sort_keys=False)
    with open(args.root_out, "w", encoding="utf-8") as f:
        f.write(base)
    if args.names_out:
        os.makedirs(os.path.dirname(args.names_out) or ".", exist_ok=True)
        with open(args.names_out, "w", encoding="utf-8") as f:
            for i, n in enumerate(names):
                f.write("%d %s\n" % (i, n))
    print("")
    print("已寫出訓練用 yaml：", args.out, "（共", len(names), "類）")
    print("資料集位置：", base)


if __name__ == "__main__":
    main()
