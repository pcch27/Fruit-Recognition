# -*- coding: utf-8 -*-
"""B2：用 YOLO 訓練多種水果偵測，訓練完印出整體成績與「每個類別」的成績。
用法（Colab，要接上 GPU）：
  python B/B2_train_yolo.py --data /content/dfv16/data.yaml --epochs 50
想比較大一點的模型：加  --weights yolov8s.pt --name b_yolov8s
注意：如果報「找不到照片」，把 data.yaml 裡的 path 改成資料夾的完整路徑。
"""
import argparse
import json
import os
import shutil


def main():
    import torch
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--name", default="b_yolo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"
        print("提醒：沒有偵測到 GPU，訓練會非常慢。請在 Colab 把執行階段改成 GPU")

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, patience=args.patience, device=device, exist_ok=True, name=args.name)
    save_dir = str(model.trainer.save_dir)
    best_path = os.path.join(save_dir, "weights", "best.pt")

    os.makedirs("models", exist_ok=True)
    final_path = os.path.join("models", "b_yolo.pt")
    shutil.copy(best_path, final_path)

    best_model = YOLO(best_path)
    names = best_model.names
    summary = {"weights": args.weights, "epochs": args.epochs, "imgsz": args.imgsz}
    for split in ["val", "test"]:
        try:
            metrics = best_model.val(data=args.data, split=split, imgsz=args.imgsz, device=device, verbose=False)
        except Exception as error:
            print(split, "評估失敗：", error)
            continue
        item = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map), "per_class": {}}
        print("")
        print("[" + split + "] 精確率", round(item["precision"], 3), " 召回率", round(item["recall"], 3),
              " mAP50", round(item["mAP50"], 3), " mAP50-95", round(item["mAP50_95"], 3))
        class_ids = list(metrics.box.ap_class_index)
        for k in range(len(class_ids)):
            class_name = names[int(class_ids[k])]
            item["per_class"][class_name] = {"AP50": float(metrics.box.ap50[k]), "AP50_95": float(metrics.box.ap[k])}
            print("   %-18s AP50 = %.3f" % (class_name, float(metrics.box.ap50[k])))
        summary[split] = item
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", args.name + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("")
    print("權重已複製到", final_path)
    print("完整訓練紀錄在", save_dir)


if __name__ == "__main__":
    main()
