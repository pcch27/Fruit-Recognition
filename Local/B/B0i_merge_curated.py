# -*- coding: utf-8 -*-
"""B0i：把「我挑好的補訓資料」fruit26_extra_dataset_curated.zip 合併進 B0f 做好的 21 類水果資料集，變成 26 類。
用法：python B/B0i_merge_curated.py --base /content/dfv_fruit --curated /content/extra/fruit26_extra
  --base      B0f 做出來的資料集（要有 data_fruit.yaml，21 類）
  --curated   解壓縮後的 fruit26_extra 資料夾（裡面有 images、labels、data_fruit.yaml，26 類）
做的事：
  1. 檢查 26 類名單的前 21 類，和 base 的 21 類名稱與順序完全相同（不同就停止，避免編號錯位）。
  2. 把 curated 的圖片與標註複製進 base 的 train／val／test。
  3. 新增類別（編號 21～25）的 train 圖片太少，重複放入（檔名加 _r0、_r1…），讓每類的有效張數接近 --target_eff，最多 --max_repeat 次。
  4. 把 base 的 data_fruit.yaml 改成 26 類。
"""
import argparse
import os
import shutil
import sys
from collections import Counter

import yaml

IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def place(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def names_of(path):
    names = yaml.safe_load(open(path, encoding="utf-8"))["names"]
    return [str(names[i]) for i in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--curated", required=True)
    ap.add_argument("--target_eff", type=int, default=80, help="新增類別希望的有效訓練張數；0 = 不重複放入")
    ap.add_argument("--max_repeat", type=int, default=6)
    ap.add_argument("--names_out", default="")
    args = ap.parse_args()

    base_yaml = os.path.join(args.base, "data_fruit.yaml")
    cur_yaml = os.path.join(args.curated, "data_fruit.yaml")
    for p in (base_yaml, cur_yaml):
        if not os.path.exists(p):
            raise SystemExit("找不到：" + p)
    base_names, cur_names = names_of(base_yaml), names_of(cur_yaml)
    print("base：%d 類；curated：%d 類" % (len(base_names), len(cur_names)))
    if len(base_names) != 21:
        raise SystemExit("base 不是 21 類（可能已經合併過；請重新執行 B0f 再合併一次）：%d 類" % len(base_names))
    if cur_names[:21] != base_names:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(base_names, cur_names[:21])) if a != b]
        raise SystemExit("前 21 類的名稱或順序不一樣，不能合併（編號會錯位）：" + str(diff[:5]))
    new_ids = set(range(21, len(cur_names)))

    # 先數：新增類別在 train 的原圖張數
    per_class = Counter()
    train_items = []
    lab_dir = os.path.join(args.curated, "labels", "train")
    for f in sorted(os.listdir(lab_dir)) if os.path.isdir(lab_dir) else []:
        ids = {int(l.split()[0]) for l in open(os.path.join(lab_dir, f), encoding="utf-8").read().splitlines() if l.strip()}
        cls = ids & new_ids
        train_items.append((os.path.splitext(f)[0], cls))
        for c in cls:
            per_class[c] += 1
    repeat = {c: (1 if args.target_eff <= 0 else max(1, min(args.max_repeat, -(-args.target_eff // max(1, per_class[c]))))) for c in per_class}
    print("新增類別的訓練原圖張數與重複次數：", {cur_names[c]: (per_class[c], repeat[c]) for c in sorted(per_class)})

    stat = Counter()
    for split in ("train", "val", "test"):
        idir, ldir = os.path.join(args.curated, "images", split), os.path.join(args.curated, "labels", split)
        if not os.path.isdir(idir):
            continue
        oi, ol = os.path.join(args.base, "images", split), os.path.join(args.base, "labels", split)
        os.makedirs(oi, exist_ok=True)
        os.makedirs(ol, exist_ok=True)
        cls_of = dict(train_items) if split == "train" else {}
        for name in sorted(os.listdir(idir)):
            if not name.endswith(IMG_EXT):
                continue
            stem, ext = os.path.splitext(name)
            lp = os.path.join(ldir, stem + ".txt")
            if not os.path.exists(lp):
                stat["缺標註"] += 1
                continue
            copies = max([repeat[c] for c in cls_of.get(stem, set())] or [1]) if split == "train" else 1
            for k in range(copies):
                suffix = "_r%d" % k if copies > 1 else ""
                place(os.path.join(idir, name), os.path.join(oi, "x_" + stem + suffix + ext))
                shutil.copy2(lp, os.path.join(ol, "x_" + stem + suffix + ".txt"))
            stat[split + " 原圖"] += 1
            stat[split + " 放入的檔案"] += copies

    config = yaml.safe_load(open(base_yaml, encoding="utf-8"))
    config["names"] = {i: n for i, n in enumerate(cur_names)}
    with open(base_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    if args.names_out:
        with open(args.names_out, "w", encoding="utf-8") as f:
            f.write("\n".join(cur_names) + "\n")
    print("\n合併結果：", dict(stat))
    print("現在共 %d 類：%s" % (len(cur_names), cur_names))
    print("yaml：", base_yaml)


if __name__ == "__main__":
    sys.exit(main())
