# -*- coding: utf-8 -*-
"""B6：在 Colab 用 YOLO 自動替照片畫框（取代網頁標註工具）。
輸出三樣東西：
  1. 畫好框的圖片        --out 資料夾（每張照片一張 jpg，框上寫「類別編號 信心」）
  2. YOLO 格式的框檔      寫在照片旁邊（同檔名 .txt；沒有偵測到就寫空檔），B5 直接讀得到
  3. 打包                 --zip_dir 有給的話，存成 b_boxes_images.zip（畫框圖）與 b_labels.zip（框檔）
注意：這些框是模型自己預測的，不是人工標註。拿它跑 B5 只是「模型和自己比」的自我一致性檢查，
      不是準確率。信心門檻預設 0.05（偏低，所以框會比較多）。
用法：
  python B/B6_auto_boxes.py --weights models/b_yolo.pt --photos /content/b_test/photos --out /content/b_test/annotated
"""
import argparse
import glob
import os
import zipfile

from PIL import Image, ImageDraw

IMAGE_END = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40)]


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--photos", required=True, help="要畫框的照片資料夾；框檔也會寫在這裡")
    parser.add_argument("--out", required=True, help="畫好框的圖片存放資料夾")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--zip_dir", default="", help="有給就把結果打包存到這個資料夾（例如 Drive）")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(p for p in glob.glob(os.path.join(args.photos, "*")) if p.lower().endswith(IMAGE_END))
    if len(paths) == 0:
        raise SystemExit("在 " + args.photos + " 找不到照片，請先執行 B4")

    model = YOLO(args.weights)
    total_boxes = 0
    label_files = []
    image_files = []
    for path in paths:
        result = model.predict(source=path, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        line_width = max(2, width // 300)
        lines = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.tolist()
            xywhn = result.boxes.xywhn.tolist()
            confs = result.boxes.conf.tolist()
            classes = result.boxes.cls.tolist()
            for i in range(len(xyxy)):
                cls = int(classes[i])
                color = COLORS[cls % len(COLORS)]
                x1, y1, x2, y2 = xyxy[i]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                draw.text((x1 + 3, y1 + 3), str(result.names[cls]) + " " + str(int(round(confs[i] * 100))) + "%", fill=color)
                cx, cy, w, h = xywhn[i]
                lines.append("%d %.6f %.6f %.6f %.6f" % (cls, cx, cy, w, h))
        total_boxes = total_boxes + len(lines)

        stem = os.path.splitext(os.path.basename(path))[0]
        label_path = os.path.join(args.photos, stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:       # 沒偵測到就是空檔，B5 才不會略過這張
            f.write("\n".join(lines) + ("\n" if lines else ""))
        image_path = os.path.join(args.out, stem + "_boxes.jpg")
        image.save(image_path, quality=92)
        label_files.append(label_path)
        image_files.append(image_path)

    print("照片", len(paths), "張，共畫了", total_boxes, "個框")
    print("畫框圖片：", args.out)
    print("框檔（YOLO 格式）：", args.photos)

    if args.zip_dir:
        os.makedirs(args.zip_dir, exist_ok=True)
        for zip_name, files in [("b_boxes_images.zip", image_files), ("b_labels.zip", label_files)]:
            zip_path = os.path.join(args.zip_dir, zip_name)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for file_path in files:
                    z.write(file_path, os.path.basename(file_path))
            print("已打包：", zip_path)


if __name__ == "__main__":
    main()
