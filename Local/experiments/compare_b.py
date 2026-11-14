# -*- coding: utf-8 -*-
"""把網頁下載的「B_YOLO數量_時間.csv」和 answers.csv 對起來，算 B 的結果。
用法（cmd，在 3_Local 資料夾）：
  python tools\\compare_b.py 網頁下載的B表.csv answers.csv
answers.csv 格式和 compare_a.py 相同：第一欄檔名（含 .jpg），第二欄答案，多種水果用 | 分隔。
統計：
  1. 有框到正確種類的圖：該圖至少有一個框的種類在答案裡（多種水果的圖算「答案被框到的比例」）
  2. 完全沒偵測到的圖
  3. 有多框其他種類的圖（框出的種類不在答案裡）
這份只用到 Python 內建套件。"""
import csv
import sys

ALIAS = {"limes": "lime", "pitahaya": "dragonfruit"}


def norm(text):
    t = text.strip().lower().replace(" ", "").replace("-", "")
    return ALIAS.get(t, t)


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def main():
    if len(sys.argv) < 3:
        raise SystemExit("用法：python tools\\compare_b.py 網頁下載的B表.csv answers.csv")
    answers = {}
    for row in read_rows(sys.argv[2])[1:]:
        if len(row) >= 2:
            answers[row[0].strip()] = [norm(x) for x in row[1].split("|") if x.strip()]

    detected = {}  # 檔名 -> 偵測到的種類集合（空集合代表沒偵測到）
    for row in read_rows(sys.argv[1])[1:]:
        if len(row) < 3:
            continue
        name, kind = row[0].strip(), row[1].strip()
        if name == "全部合計":
            continue
        s = detected.setdefault(name, set())
        if "沒有偵測" not in kind:
            s.add(norm(kind))

    s_n = s_hit = m_n = 0
    m_found = 0.0
    no_det = []
    extra = []
    skipped = 0
    for name in detected:
        if name not in answers:
            skipped = skipped + 1
            continue
        truth = answers[name]
        kinds = detected[name]
        if not kinds:
            no_det.append(name)
        if len(truth) == 1:
            s_n = s_n + 1
            s_hit = s_hit + (1 if truth[0] in kinds else 0)
        else:
            m_n = m_n + 1
            m_found = m_found + sum(1 for t in truth if t in kinds) / float(len(truth))
        if [k for k in kinds if k not in truth]:
            extra.append(name)

    print("對得上答案的圖：", s_n + m_n, "；沒對上答案的圖：", skipped)
    if s_n:
        print("單一水果 %d 張：有框到正確種類 %d 張（%.1f%%）" % (s_n, s_hit, 100.0 * s_hit / s_n))
    if m_n:
        print("多種水果 %d 張：答案種類被框到的平均比例 %.1f%%" % (m_n, 100.0 * m_found / m_n))
    print("完全沒偵測到的圖 %d 張：%s" % (len(no_det), "、".join(no_det)))
    print("有多框其他種類的圖 %d 張" % len(extra))


if __name__ == "__main__":
    main()
