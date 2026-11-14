# -*- coding: utf-8 -*-
"""B5：用「已經有框」的真實照片評估 B（YOLO），分組統計。
⚠️ 框如果是 B6 讓 YOLO 自己畫的（不是人工標註），這裡比的是「模型和自己」，只是自我一致性檢查，
   不能當成真正的準確率；類別編號也沒有水果名稱，所以「種類正確率」會是 0%。
每一組（covered / not_covered / mixed）都會算：
  找到率   ：標註的水果框，有幾成被偵測框找到（重疊比例 >= 門檻；不管種類對不對）
  精確率   ：偵測框裡有幾成對得上標註框
  種類正確率：找到的框裡，種類也對的比例（只有 covered 與 mixed 裡 B 認得的種類才有意義）
  數量誤差 ：每張照片「偵測到幾個水果」與「標註幾個」的平均絕對誤差
蔬菜類的偵測框會被忽略（因為標註時不用框蔬菜）。
用法：
  python B/B5_eval_real.py --weights models/b_yolo.pt --photos /content/b_test/photos --classes /content/b_test/classes.txt --groups /content/b_test/picked_list.csv
標註檔（YOLO 格式的 txt）預設放在照片同一個資料夾，也可以用 --labels 指定另一個資料夾。
"""
import argparse
import csv
import json
import os
import sys

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import a_common as ac
import b_eval_utils as be


def read_gt(label_path, class_names, image_w, image_h):
    """讀一張照片的標註檔，回傳 [{'family': 水果, 'box': (x1,y1,x2,y2)}, ...]。"""
    boxes = []
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            if class_id < 0 or class_id >= len(class_names):
                continue
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            boxes.append({"family": class_names[class_id], "box": be.yolo_to_xyxy(cx, cy, w, h, image_w, image_h)})
    return boxes


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--groups", required=True, help="B4 產生的 picked_list.csv")
    parser.add_argument("--labels", default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    label_dir = args.labels if args.labels != "" else args.photos

    with open(args.classes, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.read().splitlines() if line.strip() != ""]
    group_of = {}
    with open(args.groups, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            group_of[row["file"]] = row["group"]

    model = YOLO(args.weights)
    totals = {}
    no_label = []
    for name in sorted(group_of.keys()):
        photo_path = os.path.join(args.photos, name)
        if not os.path.exists(photo_path):
            continue
        image = Image.open(photo_path)
        image_w, image_h = image.size
        gt = read_gt(os.path.join(label_dir, os.path.splitext(name)[0] + ".txt"), class_names, image_w, image_h)
        if gt is None:
            no_label.append(name)
            continue
        result = model.predict(source=photo_path, conf=args.conf, verbose=False)[0]
        preds = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.tolist()
            confs = result.boxes.conf.tolist()
            classes = result.boxes.cls.tolist()
            for i in range(len(boxes)):
                pred_name = result.names[int(classes[i])]
                if not ac.is_fruit(be.family_of_name(pred_name)):
                    continue      # 蔬菜的偵測框忽略
                preds.append({"name": pred_name, "conf": float(confs[i]), "box": tuple(boxes[i])})
        matched, class_ok, false_pos, missed = be.match_boxes(gt, preds, args.iou)
        group = group_of[name]
        if group not in totals:
            totals[group] = {"images": 0, "gt": 0, "pred": 0, "matched": 0, "class_ok": 0, "count_error": 0.0}
        t = totals[group]
        t["images"] = t["images"] + 1
        t["gt"] = t["gt"] + len(gt)
        t["pred"] = t["pred"] + len(preds)
        t["matched"] = t["matched"] + matched
        t["class_ok"] = t["class_ok"] + class_ok
        t["count_error"] = t["count_error"] + abs(len(preds) - len(gt))

    if len(no_label) > 0:
        print("找不到標註檔的照片（略過）：", len(no_label), "張，例如", no_label[:3])
    print("")
    print("%-12s %5s %6s %6s %9s %9s %12s %10s" % ("組別", "照片", "標註框", "偵測框", "找到率", "精確率", "種類正確率", "數量誤差"))
    report = {}
    for group in ["covered", "not_covered", "mixed"]:
        if group not in totals:
            continue
        t = totals[group]
        recall = t["matched"] / float(max(1, t["gt"]))
        precision = t["matched"] / float(max(1, t["pred"]))
        class_rate = t["class_ok"] / float(max(1, t["matched"]))
        count_error = t["count_error"] / float(max(1, t["images"]))
        report[group] = {"images": t["images"], "gt_boxes": t["gt"], "pred_boxes": t["pred"], "recall": recall,
                         "precision": precision, "class_correct_rate": class_rate, "count_mae": count_error}
        print("%-12s %5d %6d %6d %8.1f%% %8.1f%% %11.1f%% %10.2f" % (group, t["images"], t["gt"], t["pred"],
                                                                      recall * 100, precision * 100, class_rate * 100, count_error))
    print("")
    print("怎麼讀：covered 是 B 現有能力；not_covered 預期很差（B 沒學過這些水果），要誠實寫進報告；mixed 是實際使用情境。")
    print("標註框數很少的組別，百分比不穩定，請一併寫出標註框數。")
    os.makedirs("results", exist_ok=True)
    with open("results/b_real_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
