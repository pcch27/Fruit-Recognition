# -*- coding: utf-8 -*-
"""A4：評估已經訓練好的模型。可以額外測一個資料夾的照片（例如你的 49 張圖庫照片），那些照片完全沒參與訓練。
檔名規則：'Apple_test01.jpg' -> 水果是 apple；'Grape_White_test01.jpg' -> grape（取第一個字）。

用法：
  python A/A4_evaluate.py --weights /content/models/a_mobilenetv2.pth --lists /content/lists --stock_dir /content/stock49
"""
import argparse
import csv
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import a_common as ac
import a_data_utils as du


def family_from_stock_name(file_name):
    """圖庫照片的檔名 -> 大類。取檔名第一個英文字，例如 'Grape_White_test01.jpg' -> 'grape'。"""
    stem = os.path.splitext(os.path.basename(file_name))[0].lower()
    first = stem.split("_")[0]
    if first in ac.NAME_ALIAS:
        first = ac.NAME_ALIAS[first]
    return first


def real_single_report(model, items, families, device, batch, workers, name):
    """真實照片（單一種類）：每個水果的第一名正確率、最常搞混的組合、混淆矩陣圖。"""
    singles = []
    for item in items:
        if len(item[1]) == 1:
            singles.append(item)
    if len(singles) == 0:
        return
    num_classes = len(families)
    scores = du.predict_scores(model, singles, num_classes, device, batch, workers)
    true_ids = []
    pred_ids = []
    for i in range(len(singles)):
        true_ids.append(singles[i][1][0])
        pred_ids.append(ac.top_indices(scores[i].tolist(), 1)[0])
    # 每個水果
    per_family = {}
    for t, p in zip(true_ids, pred_ids):
        if t not in per_family:
            per_family[t] = [0, 0]
        per_family[t][0] = per_family[t][0] + 1
        if t == p:
            per_family[t][1] = per_family[t][1] + 1
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", name + "_per_family.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["水果", "張數", "第一名答對", "正確率"])
        for t in sorted(per_family.keys(), key=lambda k: -per_family[k][0]):
            n, hit = per_family[t]
            writer.writerow([families[t], n, hit, round(hit / float(n), 3)])
    print("每個水果的真實照片正確率已存到", csv_path)
    # 最常搞混的組合
    confusions = {}
    for t, p in zip(true_ids, pred_ids):
        if t != p:
            confusions[(t, p)] = confusions.get((t, p), 0) + 1
    ranked = sorted(confusions.keys(), key=lambda k: -confusions[k])[:10]
    print("最常搞混的組合（真實 -> 模型說）：")
    for key in ranked:
        print("  ", families[key[0]], "->", families[key[1]], ":", confusions[key], "張")
    # 混淆矩陣圖
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        used = sorted(set(true_ids) | set(pred_ids))
        index_of = {}
        for k in range(len(used)):
            index_of[used[k]] = k
        size = len(used)
        matrix = [[0.0] * size for _ in range(size)]
        row_total = [0] * size
        for t, p in zip(true_ids, pred_ids):
            matrix[index_of[t]][index_of[p]] = matrix[index_of[t]][index_of[p]] + 1
            row_total[index_of[t]] = row_total[index_of[t]] + 1
        for r in range(size):
            if row_total[r] > 0:
                for c in range(size):
                    matrix[r][c] = matrix[r][c] / row_total[r]
        figure, axis = plt.subplots(figsize=(max(6, size * 0.4), max(5, size * 0.4)))
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        labels = [families[u] for u in used]
        axis.set_xticks(range(size))
        axis.set_yticks(range(size))
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_xlabel("model says")
        axis.set_ylabel("true")
        axis.set_title("Real photos (single fruit): confusion matrix")
        figure.tight_layout()
        png_path = os.path.join("results", name + "_confusion_real.png")
        figure.savefig(png_path, dpi=150)
        plt.close(figure)
        print("混淆矩陣圖已存到", png_path)
    except Exception as error:
        print("畫混淆矩陣失敗（不影響成績）：", error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--lists", required=True)
    parser.add_argument("--stock_dir", default="")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_eval_white", type=int, default=5000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.weights, map_location=device)
    families = checkpoint["families"]
    num_classes = len(families)
    model = ac.build_model(checkpoint["model_name"], num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    print("模型：", checkpoint["model_name"], "  大類數：", num_classes)

    with open(os.path.join(args.lists, "white_lists.json"), "r", encoding="utf-8") as f:
        white = json.load(f)
    with open(os.path.join(args.lists, "real_lists.json"), "r", encoding="utf-8") as f:
        real = json.load(f)

    results = {}
    for key in ["100_test", "original_test"]:
        items = white.get(key, [])
        if len(items) > 0:
            r = du.evaluate_white(model, items, num_classes, device, args.batch, args.workers, args.max_eval_white, 42)
            results["white_" + key] = r
            print("白底", key, ": 第一名", round(r["top1"], 4), "（", r["n"], "張）")
    if len(real.get("test", [])) > 0:
        r = du.evaluate_real(model, real["test"], num_classes, device, args.batch, args.workers)
        results["real_test"] = r
        print("真實照片 test:", r)
        report_name = os.path.splitext(os.path.basename(args.weights))[0]
        real_single_report(model, real["test"], families, device, args.batch, args.workers, report_name)

    if args.stock_dir != "":
        index_of = {}
        for i in range(num_classes):
            index_of[families[i]] = i
        items = []
        skipped = []
        for name in sorted(os.listdir(args.stock_dir)):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fam = family_from_stock_name(name)
            if fam in index_of:
                items.append([os.path.join(args.stock_dir, name), [index_of[fam]]])
            else:
                skipped.append(name)
        r = du.evaluate_real(model, items, num_classes, device, args.batch, args.workers)
        results["stock"] = r
        print("圖庫照片", len(items), "張：第一名", r.get("single_top1"), " 前三名", r.get("single_top3"))
        if len(skipped) > 0:
            print("種類不在模型大類內而略過：", skipped)

    name = os.path.splitext(os.path.basename(args.weights))[0]
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", name + "_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("成績已存到", out_path)


if __name__ == "__main__":
    main()
