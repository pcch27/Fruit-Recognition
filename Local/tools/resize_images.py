# -*- coding: utf-8 -*-
"""把測試圖縮小（網頁辨識時本來就會縮到 1280，縮小檔案不會影響辨識結果，上傳、打包會快很多）。
用法（cmd，在 3_Local 資料夾）：
  python tools\\resize_images.py 原圖資料夾 輸出資料夾 [最長邊像素，預設 1600]
例：python tools\\resize_images.py D:\\stock_test D:\\stock_test_small
- 不會動原圖；輸出一律存成 .jpg（品質 90），檔名主檔名不變（Apple_new01.png -> Apple_new01.jpg）
- 子資料夾結構會一併複製；已經比較小的圖不會放大。
只需要 Pillow。"""
import os
import sys

from PIL import Image, ImageOps

END = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main():
    if len(sys.argv) < 3:
        raise SystemExit("用法：python tools\\resize_images.py 原圖資料夾 輸出資料夾 [最長邊像素]")
    src = os.path.abspath(sys.argv[1])
    dst = os.path.abspath(sys.argv[2])
    longest = int(sys.argv[3]) if len(sys.argv) > 3 else 1600
    if dst == src or dst.startswith(src + os.sep):
        raise SystemExit("輸出資料夾不要放在原圖資料夾裡面，請換一個位置")
    count = 0
    before = 0
    after = 0
    for root, dirs, files in os.walk(src):
        for name in sorted(files):
            if not name.lower().endswith(END):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(root, src)
            out_dir = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(out_dir, exist_ok=True)
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            if max(image.size) > longest:
                image.thumbnail((longest, longest))
            out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".jpg")
            image.save(out_path, quality=90)
            count = count + 1
            before = before + os.path.getsize(path)
            after = after + os.path.getsize(out_path)
    print("處理 %d 張；原本 %.1f MB -> 現在 %.1f MB" % (count, before / 1048576.0, after / 1048576.0))
    print("輸出位置：", dst)


if __name__ == "__main__":
    main()
