# -*- coding: utf-8 -*-
"""診斷 B 為什麼漏框：同一張照片，用不同的「信心門檻」「圖片大小」「分塊」各跑一次，比較框到幾個。
用法（cmd，在 3_Local 資料夾）：
  python diagnose_b.py 照片路徑.jpg                  （一張照片，列出完整表格）
  python diagnose_b.py 照片1.jpg 照片2.jpg ...      （多張照片，只列最關鍵的 5 種設定，並列出花的秒數）
  python diagnose_b.py --auto 照片資料夾             （整個資料夾，每張一行：先用預設辨識，框得多、照片又大才分塊）
  python diagnose_b.py --variants 照片1.jpg 照片2.jpg （每張一行，比較 5 種分塊做法的框數，用來調分塊參數）
  python diagnose_b.py 照片路徑.jpg --draw 0.15,960,2     （把「信心 0.15、圖片 960、分塊 2×2」的框畫出來存成圖）
表格裡每一格是「框數（最多的種類）」。分塊＝把照片切成 n×n 塊（邊緣重疊 20%）各自辨識，再和整張的結果合併，
水果很小很密時比較找得到，但比較慢。分塊和網頁介面用同一套程式（b_infer.detect_tiled），會丟掉被切到一半的框並去掉重複的框。
這支程式只是診斷，不會改變網頁介面的任何設定。需要已安裝 ultralytics（和網頁介面一樣）。"""
import argparse
import csv
import glob
import os
import sys
import time

from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_infer

MAX_SIDE = 1280      # 和網頁介面一樣：長邊超過 1280 px 先縮小


def load_image(path):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if MAX_SIDE and max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def detect_tiled(image, conf, iou_limit, imgsz, agnostic, tiles):
    """tiles = 1：整張辨識；tiles >= 2：整張＋n×n 分塊（和網頁介面的自動分塊用同一套程式 b_infer.detect_tiled）。"""
    if tiles <= 1:
        return b_infer.detect(image, conf, iou_limit, imgsz, agnostic)
    return b_infer.detect_tiled(image, conf, iou_limit, imgsz, agnostic, n=tiles)


def summary(found):
    counts = {}
    for item in found:
        counts[item["name"]] = counts.get(item["name"], 0) + 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:2]
    text = ",".join(name + str(num) for name, num in top)
    return str(len(found)) + (" (" + text + ")" if text else "")


def draw(image, found, path):
    canvas = image.copy()
    pen = ImageDraw.Draw(canvas)
    for item in found:
        pen.rectangle(item["box"], outline=(220, 40, 90), width=3)
        pen.text((item["box"][0] + 3, item["box"][1] + 2), item["name"] + " " + str(int(item["conf"] * 100)) + "%", fill=(255, 255, 0))
    canvas.save(path, quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--iou", type=float, default=0.5, help="重疊框合併，介面預設 0.5")
    parser.add_argument("--agnostic", type=int, default=1, help="不同種類重疊也合併：1 是（介面預設），0 否")
    parser.add_argument("--variants", action="store_true", help="比較 5 種分塊做法（切邊要不要丟、切塊重疊多少、怎麼去重複）")
    parser.add_argument("--auto", action="store_true", help="自動模式：先用預設辨識，框數 >= --min_boxes 而且照片長邊 >= --min_side 才改用分塊 2×2")
    parser.add_argument("--csv", default="diagnose_auto.csv")
    parser.add_argument("--min_boxes", type=int, default=b_infer.TILE_MIN_BOXES)
    parser.add_argument("--min_side", type=int, default=b_infer.TILE_MIN_SIDE)
    parser.add_argument("--draw", default="", help="把某個設定的框畫出來，格式：信心,圖片大小,分塊，例如 0.15,960,2")
    args = parser.parse_args()

    if not b_infer.available():
        raise SystemExit("找不到 models\\b_yolo.pt")
    files = expand(args.images)
    if not files:
        raise SystemExit("找不到照片")
    if args.variants:
        variants(args, files)
        return
    if args.auto:
        auto(args, files)
        return
    if len(files) > 1:
        args.images = files
        quick(args)
        return
    full(args, files[0])


def expand(paths):
    """資料夾展開成裡面所有 jpg／png，其餘維持原樣。"""
    files = []
    for path in paths:
        if os.path.isdir(path):
            found = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                found = found + glob.glob(os.path.join(path, ext))
            files = files + sorted(set(found))
        else:
            files.append(path)
    return files


def auto(args, files):
    """每張照片一行：預設框數 → 是否分塊 → 最後框數。同時存成 diagnose_auto.csv。"""
    rows = []
    print("規則：先用預設（信心0.35、圖片640）辨識；框數 >=", args.min_boxes, "且照片長邊 >=", args.min_side, "才改用分塊 2×2")
    print("")
    for path in files:
        start = time.time()
        image = load_image(path)
        base = b_infer.detect(image, 0.35, args.iou, 640, bool(args.agnostic))
        use_tiles = len(base) >= args.min_boxes and max(image.size) >= args.min_side
        final = b_infer.detect_tiled(image, 0.35, args.iou, 640, bool(args.agnostic), first=base) if use_tiles else base
        seconds = time.time() - start
        name = os.path.basename(path)
        print(name.ljust(30) + (str(image.size[0]) + "x" + str(image.size[1])).ljust(12) + "預設 " + str(len(base)).rjust(3)
              + "  " + ("分塊" if use_tiles else "不分塊") + "  最後 " + str(len(final)).rjust(3) + "  " + summary(final).split(" ", 1)[-1].ljust(26) + str(round(seconds, 1)) + " 秒")
        rows.append([name, image.size[0], image.size[1], len(base), "是" if use_tiles else "否", len(final), summary(final), round(seconds, 1)])
    tiled = [r for r in rows if r[4] == "是"]
    print("")
    print("共", len(rows), "張；自動分塊", len(tiled), "張")
    if tiled:
        ratios = [r[5] / max(1, r[3]) for r in tiled]
        print("分塊的照片，框數平均變成原本的", round(sum(ratios) / len(ratios), 2), "倍；最大", round(max(ratios), 2), "倍")
    with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["檔名", "寬", "高", "預設框數", "是否分塊", "最後框數", "最後（最多的兩種）", "秒數"])
        writer.writerows(rows)
    print("已存：", args.csv)


VARIANTS = [
    ("不分塊", None),
    ("現行：重疊0.2 丟切邊 去重複", dict(overlap=0.2, drop_cut=True, ios_limit=b_infer.MERGE_IOS)),
    ("重疊0.5 丟切邊", dict(overlap=0.5, drop_cut=True, ios_limit=b_infer.MERGE_IOS)),
    ("重疊0.2 不丟切邊 去重複", dict(overlap=0.2, drop_cut=False, ios_limit=b_infer.MERGE_IOS)),
    ("重疊0.2 不丟切邊 只用IoU", dict(overlap=0.2, drop_cut=False, ios_limit=9.0)),
]


def variants(args, files):
    """每張照片一行，列出 5 種做法的框數（信心0.35、圖片640），數字是全部框（不分種類）。"""
    print("每張照片一行：" + "｜".join(label for label, _ in VARIANTS))
    print("")
    for path in files:
        image = load_image(path)
        base = b_infer.detect(image, 0.35, args.iou, 640, bool(args.agnostic))
        counts = []
        for label, options in VARIANTS:
            if options is None:
                counts.append(len(base))
            else:
                counts.append(len(b_infer.detect_tiled(image, 0.35, args.iou, 640, bool(args.agnostic), first=base, **options)))
        print(os.path.basename(path).ljust(28) + "  ".join(str(c).rjust(4) for c in counts))


QUICK_CONFIGS = [
    ("預設：信心0.35 圖片640", 0.35, 640, 1),
    ("圖片960", 0.35, 960, 1),
    ("圖片1280", 0.35, 1280, 1),
    ("分塊2x2 圖片640", 0.35, 640, 2),
    ("信心0.25 圖片960", 0.25, 960, 1),
]


def quick(args):
    """多張照片：每張只跑 5 種關鍵設定，列出框數（最多的兩種種類）與花的秒數。"""
    for path in args.images:
        image = load_image(path)
        print("")
        print(os.path.basename(path), "（縮成", image.size[0], "x", image.size[1], "）")
        for label, conf, size, tiles in QUICK_CONFIGS:
            start = time.time()
            found = detect_tiled(image, conf, args.iou, size, bool(args.agnostic), tiles)
            seconds = time.time() - start
            print("  " + label.ljust(22) + summary(found).ljust(28) + str(round(seconds, 1)) + " 秒")


def full(args, path):
    image = load_image(path)
    print("照片：", os.path.basename(path), "（縮成", image.size[0], "x", image.size[1], "）")
    print("每一格＝框數（最多的兩種種類）；網頁介面的預設是 信心 0.35、圖片 640、不分塊")
    confs = [0.35, 0.25, 0.15, 0.10]
    sizes = [640, 960, 1280]
    for tiles in (1, 2, 3):
        print("")
        print("【不分塊】" if tiles == 1 else "【分塊 " + str(tiles) + "×" + str(tiles) + "】")
        print("信心".ljust(8) + "".join(("圖片 " + str(s)).ljust(26) for s in sizes))
        for conf in confs:
            row = str(conf).ljust(8)
            for size in sizes:
                found = detect_tiled(image, conf, args.iou, size, bool(args.agnostic), tiles)
                row = row + summary(found).ljust(26)
            print(row)
    if args.draw:
        conf, size, tiles = args.draw.split(",")
        found = detect_tiled(image, float(conf), args.iou, int(size), bool(args.agnostic), int(tiles))
        out = "diagnose_" + os.path.splitext(os.path.basename(path))[0] + "_" + conf + "_" + size + "_t" + tiles + ".jpg"
        draw(image, found, out)
        print("")
        print("已存：", out, "（", len(found), "個框）")


if __name__ == "__main__":
    main()
