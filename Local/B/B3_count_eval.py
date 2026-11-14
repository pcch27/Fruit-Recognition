# -*- coding: utf-8 -*-
"""B3：數量評估。比較「偵測到幾個」和「人工標註幾個」（每張照片、每個類別）。
信心門檻只用 val 來選，再固定拿去測 test。
注意：這是跟標註框數比較，不是跟真實水果數比較（被遮住又沒標的水果，兩邊都不算）。
用法：
  python B/B3_count_eval.py --weights models/b_yolo.pt --root /content/dfv16
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import yolo_data as yd


def mean_abs_error(real_list, guess_list):
    """平均絕對誤差。"""
    total = 0.0
    for i in range(len(real_list)):
        total = total + abs(guess_list[i] - real_list[i])
    return total / max(1, len(real_list))


def mean_bias(real_list, guess_list):
    """平均偏差（正數代表偵測得比標註多）。"""
    total = 0.0
    for i in range(len(real_list)):
        total = total + (guess_list[i] - real_list[i])
    return total / max(1, len(real_list))


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=100, help="每批預測幾張")
    args = parser.parse_args()

    yaml_path, data = yd.find_yaml(args.root)
    names = yd.load_names(data)
    num_classes = len(names)
    model = YOLO(args.weights)

    store = {}
    for split in ["val", "test"]:
        image_dir, label_dir = yd.find_split_dirs(args.root, split)
        image_names = yd.list_images(image_dir)
        real_counts = []
        for name in image_names:
            boxes, problems = yd.read_labels(os.path.join(label_dir, os.path.splitext(name)[0] + ".txt"), num_classes)
            counts = [0] * num_classes
            for box in boxes:
                counts[box[0]] = counts[box[0]] + 1
            real_counts.append(counts)
        detections = []
        paths = []
        for name in image_names:
            paths.append(os.path.join(image_dir, name))
        # 分批預測（每批 100 張），資料集有兩萬張時才不會把記憶體吃光
        for start in range(0, len(paths), args.batch_size):
            chunk = paths[start:start + args.batch_size]
            for result in model.predict(source=chunk, conf=0.05, imgsz=args.imgsz, verbose=False, stream=True):
                if result.boxes is None or len(result.boxes) == 0:
                    detections.append(([], []))
                else:
                    detections.append((result.boxes.conf.tolist(), result.boxes.cls.tolist()))
        store[split] = (real_counts, detections)

    def guess_at(split, conf):
        guesses = []
        for scores, classes in store[split][1]:
            guesses.append(yd.counts_by_class(scores, classes, conf, num_classes))
        return guesses

    def total_error(split, conf):
        real_counts = store[split][0]
        guesses = guess_at(split, conf)
        total = 0.0
        for i in range(len(real_counts)):
            for c in range(num_classes):
                total = total + abs(guesses[i][c] - real_counts[i][c])
        return total / max(1, len(real_counts))

    best_conf = 0.25
    best_error = 1e9
    for step in range(2, 16):
        conf = step * 0.05
        e = total_error("val", conf)
        if e < best_error:
            best_error = e
            best_conf = conf
    print("在 val 上選出的信心門檻：", round(best_conf, 2))

    result_all = {"conf": best_conf}
    real_counts = store["test"][0]
    guesses = guess_at("test", best_conf)
    real_total = []
    guess_total = []
    for i in range(len(real_counts)):
        real_total.append(sum(real_counts[i]))
        guess_total.append(sum(guesses[i]))
    print("")
    print("test 每張照片的水果總數：平均絕對誤差", round(mean_abs_error(real_total, guess_total), 2),
          " 平均偏差", round(mean_bias(real_total, guess_total), 2))
    result_all["total_mae"] = mean_abs_error(real_total, guess_total)
    print("各類別（只列在 test 出現過的類別）：")
    result_all["per_class"] = {}
    for c in range(num_classes):
        real_c = []
        guess_c = []
        for i in range(len(real_counts)):
            real_c.append(real_counts[i][c])
            guess_c.append(guesses[i][c])
        if sum(real_c) == 0:
            continue
        mae = mean_abs_error(real_c, guess_c)
        bias = mean_bias(real_c, guess_c)
        result_all["per_class"][names[c]] = {"mae": mae, "bias": bias}
        print("   %-18s 平均絕對誤差 %.2f   平均偏差 %+.2f" % (names[c], mae, bias))
    os.makedirs("results", exist_ok=True)
    with open("results/b_counts.json", "w", encoding="utf-8") as f:
        json.dump(result_all, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
