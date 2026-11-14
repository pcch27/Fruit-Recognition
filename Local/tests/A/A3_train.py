# -*- coding: utf-8 -*-
"""A3：訓練一個模型。分兩個階段：
  第一階段：用白底圖訓練（100x100、original-size 或兩者都用）。ResNet18 與 MobileNetV2 凍結主幹，只訓練最後一層（和 Phase 1 相同）。
  第二階段：加入真實照片（單一種類與多種水果）一起訓練，全部參數都可以調整，並用真實照片的 val 挑最好的 epoch。
輸出是「多標籤」模型：每個水果大類各給 0~1 的分數，所以一張圖可以同時認出好幾種水果。

用法（Colab 上，三個模型各跑一次）：
  python A/A3_train.py --model mobilenetv2 --lists /content/lists --white_source both --save_dir /content/models
  python A/A3_train.py --model resnet18    --lists /content/lists --white_source both --save_dir /content/models
  python A/A3_train.py --model baseline    --lists /content/lists --white_source both --save_dir /content/models
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import a_common as ac
import a_data_utils as du


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["baseline", "resnet18", "mobilenetv2"])
    parser.add_argument("--lists", required=True, help="A2 的輸出資料夾")
    parser.add_argument("--white_source", default="both", choices=["100", "original", "both"])
    parser.add_argument("--save_dir", default="models")
    parser.add_argument("--epochs1", type=int, default=8, help="第一階段（白底）epoch 數")
    parser.add_argument("--epochs2", type=int, default=10, help="第二階段（加入真實照片）epoch 數")
    parser.add_argument("--max_per_class", type=int, default=150, help="每個 epoch 每個類別最多用幾張白底圖（越小越快）")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr1", type=float, default=0.001)
    parser.add_argument("--lr2", type=float, default=0.0001)
    parser.add_argument("--white_ratio", type=float, default=1.0, help="第二階段白底圖張數 = 真實訓練張數 x 這個數")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_eval_white", type=int, default=5000, help="白底測試集最多抽幾張來評估")
    parser.add_argument("--skip_stage2", action="store_true", help="只做第一階段")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用裝置：", device)
    if device == "cpu":
        print("提醒：沒有 GPU 會非常慢，請在 Colab 把執行階段改成 GPU")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    with open(os.path.join(args.lists, "families.json"), "r", encoding="utf-8") as f:
        families = json.load(f)
    with open(os.path.join(args.lists, "white_lists.json"), "r", encoding="utf-8") as f:
        white = json.load(f)
    with open(os.path.join(args.lists, "real_lists.json"), "r", encoding="utf-8") as f:
        real = json.load(f)
    num_classes = len(families)
    print("水果大類數：", num_classes)

    # 白底訓練清單
    keys = []
    if args.white_source in ("100", "both"):
        keys.append("100_training")
    if args.white_source in ("original", "both"):
        keys.append("original_training")
    white_items = []
    for key in keys:
        white_items = white_items + white.get(key, [])
    print("白底訓練圖：", len(white_items), "張（來源：", args.white_source, "）")
    if len(white_items) == 0:
        print("沒有白底訓練圖，請確認 A2 的路徑")
        return
    by_class = du.group_by_class(white_items)

    model = ac.build_model(args.model, num_classes, pretrained=True)
    ac.freeze_backbone(model, args.model)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # ---------------- 第一階段：白底圖 ----------------
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    optimizer = torch.optim.Adam(params, lr=args.lr1)
    history = {"stage1_loss": [], "stage2_loss": [], "stage2_val_score": []}
    start = time.time()
    fix_bn = (args.model != "baseline")   # 凍結主幹的模型，BatchNorm 也固定不更新
    for epoch in range(args.epochs1):
        items = du.pick_white(by_class, args.max_per_class, rng)
        loss = du.train_one_epoch(model, items, num_classes, optimizer, criterion, device, args.batch, args.workers, fix_bn)
        history["stage1_loss"].append(loss)
        print("第一階段 epoch", epoch + 1, "/", args.epochs1, " loss =", round(loss, 4),
              " 已用", int(time.time() - start), "秒")

    results = {"model": args.model, "white_source": args.white_source, "families": families}

    # 第一階段結束先測白底（用最後一個 epoch，不用測試集挑模型）
    print("")
    print("== 第一階段結束：白底測試 ==")
    results["stage1_white"] = {}
    for key in ["100_test", "original_test"]:
        items = white.get(key, [])
        if len(items) == 0:
            continue
        r = du.evaluate_white(model, items, num_classes, device, args.batch, args.workers, args.max_eval_white, args.seed)
        results["stage1_white"][key] = r
        print(key, ": 張數", r["n"], " 第一名正確率", round(r["top1"], 4))
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({"model_name": args.model, "families": families, "state_dict": model.state_dict()},
               os.path.join(args.save_dir, "a_" + args.model + "_stage1.pth"))

    # ---------------- 第二階段：加入真實照片 ----------------
    real_train = real.get("train", [])
    real_val = real.get("val", [])
    real_test = real.get("test", [])
    if args.skip_stage2 or len(real_train) == 0:
        print("略過第二階段（沒有真實照片或指定跳過）")
    else:
        print("")
        print("== 第二階段：白底加真實照片 ==  真實訓練", len(real_train), "張，val", len(real_val), "張")
        ac.unfreeze_all(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr2)
        best_score = -1.0
        best_state = None
        best_epoch = 0
        for epoch in range(args.epochs2):
            items = []
            for path, ids in real_train:
                items.append([path, ids, "real"])
            how_many = int(len(real_train) * args.white_ratio)
            items = items + du.pick_white_n(by_class, how_many, rng)
            loss = du.train_one_epoch(model, items, num_classes, optimizer, criterion, device, args.batch, args.workers)
            val = du.evaluate_real(model, real_val, num_classes, device, args.batch, args.workers)
            score = du.val_score(val)
            history["stage2_loss"].append(loss)
            history["stage2_val_score"].append(score)
            print("第二階段 epoch", epoch + 1, "/", args.epochs2, " loss =", round(loss, 4), " 真實 val 分數 =", round(score, 4))
            if score >= best_score:
                best_score = score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
        if best_state is not None:
            model.load_state_dict(best_state)
            print("使用 val 分數最高的 epoch：", best_epoch)
            results["stage2_best_epoch"] = best_epoch

    # ---------------- 最後評估 ----------------
    print("")
    print("== 最後評估（測試集只在這裡用一次）==")
    results["final_white"] = {}
    for key in ["100_test", "original_test"]:
        items = white.get(key, [])
        if len(items) == 0:
            continue
        r = du.evaluate_white(model, items, num_classes, device, args.batch, args.workers, args.max_eval_white, args.seed)
        results["final_white"][key] = r
        print("白底", key, ": 第一名正確率", round(r["top1"], 4), "（", r["n"], "張）")
    if len(real_test) > 0:
        r = du.evaluate_real(model, real_test, num_classes, device, args.batch, args.workers)
        results["final_real_test"] = r
        print("真實照片 test:", r)

    final_path = os.path.join(args.save_dir, "a_" + args.model + ".pth")
    torch.save({"model_name": args.model, "families": families, "state_dict": model.state_dict()}, final_path)
    os.makedirs("results", exist_ok=True)
    result_path = os.path.join("results", "a_" + args.model + "_metrics.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # 訓練曲線（存成圖片與 json，簡報可以用）
    history_path = os.path.join("results", "a_" + args.model + "_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(range(1, len(history["stage1_loss"]) + 1), history["stage1_loss"], marker="o")
        axes[0].set_title("Stage 1 (white background) loss")
        axes[0].set_xlabel("epoch")
        if len(history["stage2_loss"]) > 0:
            epochs2 = range(1, len(history["stage2_loss"]) + 1)
            axes[1].plot(epochs2, history["stage2_loss"], marker="o", label="train loss")
            axes[1].plot(epochs2, history["stage2_val_score"], marker="s", label="real val score")
            axes[1].legend()
        axes[1].set_title("Stage 2 (white + real photos)")
        axes[1].set_xlabel("epoch")
        figure.tight_layout()
        figure.savefig(os.path.join("results", "a_" + args.model + "_curves.png"), dpi=150)
        plt.close(figure)
        print("訓練曲線已存到 results/a_" + args.model + "_curves.png")
    except Exception as error:
        print("畫曲線圖失敗（不影響訓練結果）：", error)
    print("")
    print("權重已存到", final_path)
    print("成績已存到", result_path)


if __name__ == "__main__":
    main()
