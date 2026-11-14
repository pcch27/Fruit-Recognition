# -*- coding: utf-8 -*-
"""A2：建立訓練用的清單。
做的事：
  1. 掃描 Fruits-360 的 100x100 與 original-size 資料夾，把類別名稱換成「水果大類」，不要蔬菜與堅果
  2. 把 A1 整理好的真實照片，換成同一套大類編號
  3. 輸出：families.json（大類清單）、white_lists.json（白底圖清單）、real_lists.json（真實照片清單）
  4. 印出每個大類有多少白底圖、多少真實照片

用法：
  python A/A2_build_lists.py --fruits100 /content/data/fruits-360_100x100/fruits-360 \
      --original /content/data/fruits-360_original-size/fruits-360-original-size \
      --real_dir /content/real --out /content/lists
（--original 可以不給，就只用 100x100）
"""
import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import a_common as ac

IMAGE_END = (".jpg", ".jpeg", ".png")


def scan_split(root, split_name):
    """掃描 root/split_name 底下每個類別資料夾。回傳 {類別名稱: [照片路徑, ...]}。找不到資料夾就回傳空的。"""
    folder = os.path.join(root, split_name)
    result = {}
    if not os.path.isdir(folder):
        return result
    for class_name in sorted(os.listdir(folder)):
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = []
        for name in sorted(os.listdir(class_dir)):
            if name.lower().endswith(IMAGE_END):
                files.append(os.path.join(class_dir, name))
        result[class_name] = files
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fruits100", required=True, help="100x100 版本的資料夾（裡面有 Training 與 Test）")
    parser.add_argument("--original", default="", help="original-size 版本的資料夾（裡面有 Training、Validation、Test）")
    parser.add_argument("--real_dir", required=True, help="A1 的輸出資料夾（裡面有 images 與 real_manifest.csv）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 要處理的來源與資料組
    sources = [("100", args.fruits100, ["Training", "Test"])]
    if args.original != "":
        sources.append(("original", args.original, ["Training", "Validation", "Test"]))

    # 第一步：掃描，同時收集所有水果大類
    scanned = {}
    families_set = set()
    for source_name, root, split_names in sources:
        for split_name in split_names:
            data = scan_split(root, split_name)
            scanned[source_name + "_" + split_name.lower()] = data
            print("掃描", source_name, split_name, ":", len(data), "個類別")
            for class_name in data.keys():
                fam = ac.family_from_class_name(class_name)
                if ac.is_fruit(fam):
                    families_set.add(fam)
    families = sorted(families_set)
    index_of = {}
    for i in range(len(families)):
        index_of[families[i]] = i
    print("水果大類共", len(families), "種")

    # 第二步：寫白底清單
    white = {}
    white_count = {}
    for key, data in scanned.items():
        items = []
        for class_name, files in data.items():
            fam = ac.family_from_class_name(class_name)
            if fam not in index_of:
                continue
            for path in files:
                items.append([path, index_of[fam]])
            if key.endswith("_training"):
                src = key.split("_")[0]
                white_count[(src, fam)] = white_count.get((src, fam), 0) + len(files)
        white[key] = items
        print("  清單", key, ":", len(items), "張")

    # 第三步：真實照片清單
    manifest_path = os.path.join(args.real_dir, "real_manifest.csv")
    real = {"train": [], "val": [], "test": []}
    real_count = {}
    dropped = 0
    dropped_names = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names = row["families"].split("|")
                ids = []
                for n in names:
                    if n in index_of:
                        ids.append(index_of[n])
                    else:
                        dropped_names[n] = dropped_names.get(n, 0) + 1
                if len(ids) == 0:
                    dropped = dropped + 1
                    continue
                path = os.path.join(args.real_dir, "images", row["file"])
                real[row["split"]].append([path, ids])
                key = names[0] if len(names) == 1 else "multi"
                if key not in real_count:
                    real_count[key] = [0, 0, 0]
                real_count[key][["train", "val", "test"].index(row["split"])] += 1
    else:
        print("找不到", manifest_path, "（沒有真實照片，只能做第一階段）")
    print("真實照片：train", len(real["train"]), "/ val", len(real["val"]), "/ test", len(real["test"]),
          "；因為種類不在 Fruits-360 大類而略過", dropped, "張")
    if len(dropped_names) > 0:
        print("  檔名中不在 Fruits-360 水果大類的字（已忽略）：", sorted(dropped_names.keys()))

    # 寫檔
    with open(os.path.join(args.out, "families.json"), "w", encoding="utf-8") as f:
        json.dump(families, f, ensure_ascii=False)
    with open(os.path.join(args.out, "white_lists.json"), "w", encoding="utf-8") as f:
        json.dump(white, f)
    with open(os.path.join(args.out, "real_lists.json"), "w", encoding="utf-8") as f:
        json.dump(real, f)

    # 報告表
    print("")
    print("大類   白底100訓練  白底original訓練  真實單一(train/val/test)")
    for fam in families:
        w100 = white_count.get(("100", fam), 0)
        wori = white_count.get(("original", fam), 0)
        r = real_count.get(fam, [0, 0, 0])
        mark = ""
        if sum(r) == 0:
            mark = "   <- 沒有真實照片"
        print("%-14s %8d %12d        %d / %d / %d%s" % (fam, w100, wori, r[0], r[1], r[2], mark))
    if "multi" in real_count:
        print("多種水果的圖（multi）：train / val / test =", real_count["multi"])
    print("")
    print("完成，清單在", args.out)


if __name__ == "__main__":
    main()
