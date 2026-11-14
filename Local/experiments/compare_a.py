# -*- coding: utf-8 -*-
"""把網頁下載的「A_三模型判斷_時間.csv」和 real_test109 的 answers.csv 對起來，算 A 的正確率。
新舊兩種 A 表都能讀：新版有 Top 1、Top 2、Top 3 三欄，舊版只有 Top 1 和 Top 3。
用法（cmd，在 3_Local 資料夾）：
  python tools\\compare_a.py 網頁下載的A表.csv answers.csv
單一水果的圖：算 Top 1 答對、正確答案在 Top 3 的比例。
多種水果的圖（正確答案有 | 分隔）：算「正確答案有幾成出現在 Top 3」。
這份只用到 Python 內建套件。"""
import csv
import sys


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def first_word(text):
    """'lemon 67%（低信心）' -> 'lemon'"""
    return text.strip().split(" ")[0].strip()


def main():
    if len(sys.argv) < 3:
        raise SystemExit("用法：python tools\\compare_a.py 網頁下載的A表.csv answers.csv")
    all_rows = read_rows(sys.argv[1])
    header = all_rows[0] if all_rows else []
    new_format = ("Top 2" in header)      # 新版介面：Top 1、Top 2、Top 3 各一欄；舊版只有 Top 1 和 Top 3（前三名合在一欄）
    a_rows = all_rows[1:]
    answers = {}
    for row in read_rows(sys.argv[2])[1:]:
        if len(row) >= 2:
            answers[row[0].strip()] = [x.strip() for x in row[1].split("|") if x.strip()]

    stats = {}
    skipped = 0
    for row in a_rows:
        if len(row) < 4:
            continue
        if new_format:
            if len(row) < 5:
                continue
            name, model, top1 = row[0].strip(), row[1].strip(), row[2]
            top3_names = [first_word(x) for x in row[2:5] if x.strip() and x.strip() != "-"]
        else:
            name, model, top1, top3 = row[0].strip(), row[1].strip(), row[2], row[3]
            top3_names = [first_word(x) for x in top3.split("、") if x.strip()]
        if name not in answers:
            skipped = skipped + 1
            continue
        truth = answers[name]
        s = stats.setdefault(model, {"s_n": 0, "s_top1": 0, "s_top3": 0, "m_n": 0, "m_found": 0.0})
        if len(truth) == 1:
            s["s_n"] = s["s_n"] + 1
            s["s_top1"] = s["s_top1"] + (1 if first_word(top1) == truth[0] else 0)
            s["s_top3"] = s["s_top3"] + (1 if truth[0] in top3_names else 0)
        else:
            s["m_n"] = s["m_n"] + 1
            s["m_found"] = s["m_found"] + sum(1 for t in truth if t in top3_names) / float(len(truth))

    print("對得上答案的圖：以模型為單位統計；沒對上答案的列（例如不是 real_test109 的圖）：", skipped)
    for model in stats:
        s = stats[model]
        line = model + "："
        if s["s_n"]:
            line += " 單一水果 %d 張 Top1 %.1f%%  Top3 %.1f%%；" % (s["s_n"], 100.0 * s["s_top1"] / s["s_n"], 100.0 * s["s_top3"] / s["s_n"])
        if s["m_n"]:
            line += " 多種水果 %d 張，正確答案出現在 Top3 的平均比例 %.1f%%" % (s["m_n"], 100.0 * s["m_found"] / s["m_n"])
        print(line)


if __name__ == "__main__":
    main()
