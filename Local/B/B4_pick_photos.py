# -*- coding: utf-8 -*-
"""B4：從 A1 整理好的真實照片裡，挑出一批「要畫框」的照片，當 B 的獨立真實照片測試集。
挑選原則：
  - 只從 A1 的 test 資料組挑（train 留給訓練，這批照片不會拿去訓練 B）
  - 分三組，每組各挑一些（種類盡量分散，不讓蘋果佔滿）：
      covered      ：單一種類，而且種類在 DFV16 的類別裡（評估 B 現有的能力）
      not_covered  ：單一種類，但種類不在 DFV16 裡（評估 B 的限制，預期會很差，要誠實呈現）
      mixed        ：多種水果混在一起（評估實際使用情境）
輸出：photos 資料夾（要畫框的照片）、picked_list.csv（檔名、組別、水果）、classes.txt（照片檔名裡的水果清單）
注意：下面的 covered / not_covered 是照「原計畫的 DFV16 十五類名單」分的。目前實際訓練 B 用的是另一個資料集
（Zenodo 18904638，類別名稱在訓練時遺失，只剩 0~15 的編號），所以這個分組只能當參考，不代表 B 真的認得／不認得。
用法：
  python B/B4_pick_photos.py --real_dir /content/real --out /content/b_test
"""
import argparse
import csv
import os
import random
import shutil

# DFV16 的類別第一個字（水果加蔬菜）。這裡只用來判斷「照片裡的水果 B 認不認得」。
DFV16_FAMILIES = ["apple", "avocado", "banana", "corn", "cucumber", "eggplant", "kiwi", "lemon",
                  "mango", "orange", "pear", "pepper", "pomegranate", "potato", "tomato"]


def spread_pick(rows, how_many, per_family, rng):
    """從 rows 裡挑 how_many 張，每種水果先各挑 per_family 張，不夠再補。"""
    by_family = {}
    for row in rows:
        key = row["families"]
        if key not in by_family:
            by_family[key] = []
        by_family[key].append(row)
    for key in by_family:
        rng.shuffle(by_family[key])
    picked = []
    round_number = 0
    while len(picked) < how_many:
        added = False
        for key in sorted(by_family.keys()):
            start = round_number * per_family
            for row in by_family[key][start:start + per_family]:
                if len(picked) < how_many:
                    picked.append(row)
                    added = True
        round_number = round_number + 1
        if not added:
            break
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True, help="A1 的輸出資料夾（有 images 與 real_manifest.csv）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test", help="從哪個資料組挑（預設 test）")
    parser.add_argument("--n_covered", type=int, default=10)
    parser.add_argument("--n_not_covered", type=int, default=10)
    parser.add_argument("--n_mixed", type=int, default=10)
    parser.add_argument("--per_family", type=int, default=2, help="同一組裡，每種水果每輪最多挑幾張")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    with open(os.path.join(args.real_dir, "real_manifest.csv"), "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == args.split:
                rows.append(row)
    print("來源資料組", args.split, "共", len(rows), "張")

    covered = []
    not_covered = []
    mixed = []
    for row in rows:
        names = row["families"].split("|")
        if len(names) >= 2:
            mixed.append(row)
        elif names[0] in DFV16_FAMILIES:
            covered.append(row)
        else:
            not_covered.append(row)
    print("可挑的照片：covered", len(covered), "/ not_covered", len(not_covered), "/ mixed", len(mixed))

    chosen = []
    for group, pool, quota in [("covered", covered, args.n_covered),
                               ("not_covered", not_covered, args.n_not_covered),
                               ("mixed", mixed, args.n_mixed)]:
        picked = spread_pick(pool, quota, args.per_family, rng)
        if len(picked) < quota:
            print("提醒：", group, "只有", len(picked), "張（想要", quota, "張），因為", args.split, "資料組裡這類照片不夠")
        for row in picked:
            chosen.append((group, row))

    photo_dir = os.path.join(args.out, "photos")
    os.makedirs(photo_dir, exist_ok=True)
    family_set = set()
    with open(os.path.join(args.out, "picked_list.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "group", "families"])
        for group, row in chosen:
            shutil.copy(os.path.join(args.real_dir, "images", row["file"]), os.path.join(photo_dir, row["file"]))
            writer.writerow([row["file"], group, row["families"]])
            for name in row["families"].split("|"):
                family_set.add(name)
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        for name in sorted(family_set):
            f.write(name + "\n")

    print("")
    print("已挑出", len(chosen), "張，存到", photo_dir)
    print("標註時用的類別清單：", sorted(family_set), "（存在 classes.txt）")
    print("")
    print("接下來：用 B/B6_auto_boxes.py 在 Colab 讓 YOLO 自動畫框（得到畫框圖與 YOLO 格式的框檔），")
    print("再用 B5_eval_real.py 做自我一致性檢查。")
    print("這批照片只拿來測試，不要拿去訓練 B。")


if __name__ == "__main__":
    main()
