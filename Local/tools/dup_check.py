# -*- coding: utf-8 -*-
"""挑測試圖前，檢查資料夾裡有沒有「長得幾乎一樣」的圖（同一組照片、同一個攝影師的連拍）。
用法：python tools/dup_check.py 圖片資料夾 [門檻，預設 6，越小越嚴格]
原理：把每張圖縮成 9x8 灰階，算「相鄰像素亮度比較」的 64 位元指紋（dHash），指紋差異很小就算相似。
只需要 Pillow。結果只是提醒，最後要自己看一眼。"""
import os
import sys

from PIL import Image

END = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def dhash(path):
    image = Image.open(path).convert("L").resize((9, 8))
    pixels = list(image.get_flattened_data() if hasattr(image, "get_flattened_data") else image.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if pixels[row * 9 + col] > pixels[row * 9 + col + 1] else 0)
    return bits


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法：python tools/dup_check.py 圖片資料夾 [門檻]")
    folder = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(END))
    hashes = [(f, dhash(os.path.join(folder, f))) for f in files]
    found = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            diff = bin(hashes[i][1] ^ hashes[j][1]).count("1")
            if diff <= limit:
                print("很像（差 %d）：%s  <->  %s" % (diff, hashes[i][0], hashes[j][0]))
                found += 1
    print("共檢查", len(files), "張，找到", found, "組相似的圖")


if __name__ == "__main__":
    main()
