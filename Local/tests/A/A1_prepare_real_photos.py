# -*- coding: utf-8 -*-
"""A1：整理 Fruits-360 的 multi 真實照片。
做的事：
  1. 找出資料夾裡所有照片，從檔名讀出裡面有哪些水果（只留水果，蔬菜堅果不要）
  2. 把大圖縮小（長邊 448），存到新資料夾，之後訓練比較快
  3. 分成 train / val / test：同一個地點連拍的照片（連號）一定分在同一邊，避免偷看答案
  4. 輸出 real_manifest.csv（每張圖的水果、分組、資料組別）

用法（Colab 上）：
  python A/A1_prepare_real_photos.py --src /content/data/fruits-360_multi/test-multiple_fruits --out /content/real
"""
import argparse
import csv
import os
import sys

from PIL import Image, ImageOps

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import a_common as ac

IMAGE_END = (".jpg", ".jpeg", ".png")


def find_images(folder):
    """走訪資料夾，找出所有照片（不分大小寫），略過 list.txt 這類非照片檔。"""
    found = []
    for root, dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(IMAGE_END):
                found.append(os.path.join(root, name))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="test-multiple_fruits 資料夾（七份 zip 解壓縮在一起）")
    parser.add_argument("--out", required=True, help="輸出資料夾")
    parser.add_argument("--max_side", type=int, default=448, help="縮圖後的長邊")
    parser.add_argument("--block", type=int, default=8, help="連號幾張算同一組")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = find_images(args.src)
    print("找到照片：", len(paths), "張")
    if len(paths) == 0:
        print("沒有找到照片，請確認 --src 路徑")
        return

    image_dir = os.path.join(args.out, "images")
    os.makedirs(image_dir, exist_ok=True)

    rows = []        # 每一列：[新檔名, 水果清單, 組名稱, 層名稱]
    skipped = 0
    used_names = {}
    for path in paths:
        name = os.path.basename(path)
        tokens = ac.tokens_from_filename(name)
        fruits = []
        for t in tokens:
            if ac.is_fruit(t):
                fruits.append(t)
        if len(fruits) == 0:
            skipped = skipped + 1
            continue
        # 分組：只有一種水果 -> 依連號分塊；多種水果 -> 同樣組合算同一組
        if len(fruits) == 1:
            number = ac.photo_number(name)
            group = fruits[0] + "|" + str(number // args.block)
            stratum = fruits[0]
        else:
            group = ac.composition_key(name)
            stratum = "multi"
        # 縮圖並存檔
        base = os.path.splitext(name)[0].replace(" ", "_")
        new_name = base + ".jpg"
        count = used_names.get(new_name, 0)
        used_names[new_name] = count + 1
        if count > 0:
            new_name = base + "_dup" + str(count) + ".jpg"
        image = Image.open(path)
        image.draft("RGB", (args.max_side * 2, args.max_side * 2))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((args.max_side, args.max_side))
        image.save(os.path.join(image_dir, new_name), quality=90)
        rows.append([new_name, fruits, group, stratum])

    print("沒有水果（只有蔬菜或堅果）而略過：", skipped, "張")

    # 分資料組
    groups_by_stratum = {}
    for row in rows:
        stratum = row[3]
        if stratum not in groups_by_stratum:
            groups_by_stratum[stratum] = []
        if row[2] not in groups_by_stratum[stratum]:
            groups_by_stratum[stratum].append(row[2])
    split_of_group = ac.split_groups(groups_by_stratum, args.seed)

    manifest_path = os.path.join(args.out, "real_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "families", "group", "split"])
        for row in rows:
            writer.writerow([row[0], "|".join(row[1]), row[2], split_of_group[row[2]]])

    # 統計
    count_split = {}
    count_family = {}
    for row in rows:
        s = split_of_group[row[2]]
        count_split[s] = count_split.get(s, 0) + 1
        key = row[3]
        if key not in count_family:
            count_family[key] = {"train": 0, "val": 0, "test": 0}
        count_family[key][s] = count_family[key][s] + 1
    print("")
    print("各資料組張數：", count_split)
    print("")
    print("各種類（單一水果）或 multi 的 train / val / test 張數：")
    for key in sorted(count_family.keys(), key=lambda k: -sum(count_family[k].values())):
        c = count_family[key]
        total = c["train"] + c["val"] + c["test"]
        warn = ""
        if total < 10:
            warn = "   <- 張數太少，結果不穩定"
        print("  %-14s %4d   (%d / %d / %d)%s" % (key, total, c["train"], c["val"], c["test"], warn))
    print("")
    print("完成。照片在", image_dir, "，清單在", manifest_path)


if __name__ == "__main__":
    main()
