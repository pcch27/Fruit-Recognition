# -*- coding: utf-8 -*-
"""
水果辨識網頁（多張照片、顯示檔名、可下載 CSV / Excel；本機與 Hugging Face 共用）。
  A：三個分類模型（MobileNetV2、ResNet18、Baseline CNN）判斷每張照片中的水果大類。
  B：YOLO 把每張照片中偵測到的水果框出來並統計數量（多張時另有全部合計）。
本機執行：    python app.py        →  用瀏覽器開 http://127.0.0.1:7860
手機同 WiFi： Windows 命令列先輸入 set LAN=1 再執行 python app.py
Hugging Face：SDK 選 Gradio（ZeroGPU 或 CPU 都可以；有 spaces 套件才會啟用 GPU 裝飾器）。
"""

import csv
import os
import tempfile
import time
import zipfile

import gradio as gr
from PIL import Image, ImageDraw, ImageOps

# ZeroGPU：本機或 CPU 沒有 spaces 套件時，自動當成一般函式
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

import a_infer
import b_infer


BASE = os.path.dirname(os.path.abspath(__file__))

# ---------- 可以自己調整的數字 ----------
A_LOW_CONF = 0.5         # A：第一名分數低於這個值，Top 1 後面會加註「（低信心）」；不影響判斷，只是提醒
B_CONFIDENCE = 0.35      # B：YOLO 信心超過這個值才畫框
B_IOU = 0.5              # B：重疊框合併門檻（越小越容易把重疊框合併掉）
B_IMGSZ = 640            # B：推論圖片大小
B_MIN_AREA = 0.001       # B：框面積小於整張圖的這個比例就丟掉
MAX_SIDE = 1280          # 照片最長邊超過這個像素就先縮小（CPU 會快很多）；設 0 代表不縮
SECONDS_PER_IMAGE = 4    # 只有 ZeroGPU 用：預估每張照片要幾秒 GPU
BASE_SECONDS = 15        # 只有 ZeroGPU 用：固定開銷
MAX_DURATION = 120       # 只有 ZeroGPU 用：單次 GPU 呼叫上限（秒）

# ---------- 介面 ----------
# 畫面只有「上傳照片、辨識」。所有參數都固定用上面的預設值，不給使用者調。

a_infer.load_models(os.path.join(BASE, "models"))

NOTE = (
    "說明：這是學習用的 Demo。A 與 B 都只認得訓練時看過的水果；沒看過的種類可能漏掉或認錯，"
    "分數不等於實際正確率。B 的數量是「畫面中偵測到的框數」，被遮住的水果可能漏算。"
)

COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40)]

B_HEADERS = ["檔名", "種類", "數量", "平均信心"]
NO_NAME = "水果（B 無名稱）"
A_HEADERS = ["檔名", "模型", "Top 1", "Top 2", "Top 3"]


def percent(value):
    """0.8734 -> '87%'"""
    return str(int(round(float(value) * 100))) + "%"


def b_label(name):
    """YOLO 的類別名稱如果只是數字，就顯示成『類別N』，避免看起來像數量。"""
    name = str(name)
    return "類別" + name if name.isdigit() else name


def load_item(item):
    """Gallery 傳進來的項目 → (檔名, PIL 圖)。type='filepath' 時可取得原始檔名。"""
    caption = None
    if isinstance(item, (tuple, list)):
        caption = item[1] if len(item) > 1 else None
        item = item[0]
    if isinstance(item, str):
        filename = os.path.basename(item)
        image = Image.open(item)
    else:
        filename = caption or "未命名"
        image = item
    image = ImageOps.exif_transpose(image).convert("RGB")
    if MAX_SIDE and max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE))
    return filename, image


def analyze_a(image, names):
    """A：回傳這張圖的表格列（每個模型一列）：[模型, Top 1, Top 2, Top 3]。
    每一欄是一個名次的水果與分數；Top 1 的分數低於 A_LOW_CONF 時加註「（低信心）」。
    不設「有／沒有」的門檻：模型一定會給出最接近的答案，要不要相信請看分數。"""
    rows_out = []
    result = a_infer.predict(image, 5)
    for model_name in names:
        rows = result.get(model_name, [])
        cells = []
        for rank in range(3):
            if rank < len(rows):
                text = rows[rank][0] + " " + percent(rows[rank][1])
                if rank == 0 and rows[rank][1] < A_LOW_CONF:
                    text = text + "（低信心）"
            else:
                text = "-"
            cells.append(text)
        rows_out.append([model_name] + cells)
    return rows_out


# ---------- B 認得的類別、B 不認得的類別，分開處理 ----------
# B 認得的類別名稱：優先從目前載入的 B 權重讀取；讀不到時才用下面的 26 類名單。
B_CLASSES_26 = ["Apple", "Apricot", "Banana", "Cherry", "DragonFruit", "Durian", "Fig", "Grape", "Lemon", "Lime", "Lychee", "Mango",
                "Orange", "Peach", "Pear", "Pineapple", "Plum", "Pomegranate", "Quince", "Strawberry", "Watermelon",
                "Kiwi", "Raspberry", "Blackberry", "Cantaloupe", "Jackfruit"]


def _b_class_names():
    try:
        if b_infer.available():
            names = [str(n) for n in b_infer.class_names()]
            if names and not all(n.isdigit() for n in names):
                return names
    except Exception:
        pass
    return list(B_CLASSES_26)


B_CLASSES = _b_class_names()


# ---------- 中英對照（只用在畫面上的表格與說明；匯出的 CSV／Excel 和框圖上的字維持英文） ----------
ZH_NAMES = {
    "apple": "蘋果", "apricot": "杏桃", "avocado": "酪梨", "banana": "香蕉", "blackberry": "黑莓", "blueberry": "藍莓",
    "cactus": "仙人掌果", "cantaloupe": "哈密瓜", "carambola": "楊桃", "cherimoya": "番荔枝", "cherry": "櫻桃",
    "clementine": "小柑橘", "cocos": "椰子", "dates": "椰棗", "fig": "無花果", "gooseberry": "醋栗", "granadilla": "甜百香果",
    "grape": "葡萄", "grapefruit": "葡萄柚", "guava": "芭樂", "huckleberry": "越橘", "kaki": "柿子", "kiwi": "奇異果",
    "kumquats": "金桔", "lemon": "檸檬", "limes": "萊姆", "lime": "萊姆", "lychee": "荔枝", "mandarine": "柑橘", "mango": "芒果",
    "mangostan": "山竹", "maracuja": "百香果", "melon": "甜瓜", "mulberry": "桑葚", "nectarine": "油桃", "orange": "橘子",
    "papaya": "木瓜", "passion": "百香果", "peach": "桃子", "pear": "梨子", "pepino": "香瓜茄", "physalis": "酸漿果",
    "pineapple": "鳳梨", "pitahaya": "火龍果", "dragonfruit": "火龍果", "plum": "李子", "pomegranate": "石榴", "pomelo": "柚子",
    "quince": "榲桲", "rambutan": "紅毛丹", "raspberry": "覆盆子", "redcurrant": "紅醋栗", "salak": "蛇皮果",
    "strawberry": "草莓", "tamarillo": "樹番茄", "tangelo": "橘柚", "watermelon": "西瓜",
    "durian": "榴槤", "jackfruit": "菠蘿蜜",
}


def bi(name):
    """英文名稱後面加中文：cherry -> cherry（櫻桃）；查不到就維持原樣。"""
    zh = ZH_NAMES.get(str(name).strip().lower())
    return str(name) + "（" + zh + "）" if zh else str(name)


def show_b(rows):
    """B 表格的畫面版：第 2 欄（種類）加中文。"""
    return [[r[0], bi(r[1])] + list(r[2:]) if len(r) > 2 else r for r in rows]


def show_a(rows):
    """A 表格的畫面版：Top 1～3 的水果名稱加中文（例：cherry 83% -> cherry（櫻桃） 83%）。"""
    out = []
    for r in rows:
        new = list(r)
        for i in range(2, len(new)):
            parts = str(new[i]).split(" ", 1)
            if len(parts) == 2 and parts[0].replace("_", "").isalpha():
                new[i] = bi(parts[0]) + " " + parts[1]
        out.append(new)
    return out


def analyze_b(image, conf, iou, imgsz, agnostic, min_area, numeric_b=False):
    """B：回傳 (畫好框的圖, 這張圖的統計 {種類: [信心...]})。
    min_area：框面積占整張圖的比例小於這個值就丟掉（濾掉極小的雜框）。"""
    found_boxes = b_infer.detect(image, conf, iou, imgsz, agnostic)
    width, height = image.size
    image_area = float(width * height)
    b_image = image.copy()
    draw = ImageDraw.Draw(b_image)
    color_of = {}
    stats = {}
    line_width = max(2, width // 300)

    for item in found_boxes:
        x1, y1, x2, y2 = item["box"]
        if (x2 - x1) * (y2 - y1) < min_area * image_area:
            continue

        fruit_name = NO_NAME if numeric_b else b_label(item["name"])   # B 只有編號時，編號沒有意義，合併成一類
        if fruit_name not in color_of:
            color_of[fruit_name] = COLORS[len(color_of) % len(COLORS)]
        stats.setdefault(fruit_name, []).append(float(item["conf"]))
        draw.rectangle([x1, y1, x2, y2], outline=color_of[fruit_name], width=line_width)
        draw.text((x1 + 3, y1 + 3), fruit_name + " " + percent(item["conf"]), fill=color_of[fruit_name])
    return b_image, stats


def stats_rows(label, stats):
    rows = []
    for fruit_name in sorted(stats.keys(), key=lambda k: -len(stats[k])):
        confs = stats[fruit_name]
        rows.append([label, fruit_name, str(len(confs)), percent(sum(confs) / len(confs))])
    return rows


# ---------- 匯出 CSV / Excel ----------
def write_csv(path, headers, rows):
    """UTF-8 with BOM，Excel 直接開啟中文不會亂碼。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_excel(path, b_rows, a_rows):
    """寫成 .xlsx（兩個工作表）；沒有 openpyxl 時回傳 False。"""
    try:
        from openpyxl import Workbook
    except ImportError:
        return False
    wb = Workbook()
    ws_b = wb.active
    ws_b.title = "B_YOLO數量"
    ws_b.append(B_HEADERS)
    for row in b_rows:
        ws_b.append(row)
    ws_a = wb.create_sheet("A_三模型判斷")
    ws_a.append(A_HEADERS)
    for row in a_rows:
        ws_a.append(row)
    wb.save(path)
    return True


def export_files(b_rows, a_rows, jpg_paths=()):
    folder = tempfile.mkdtemp()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    paths = []
    if jpg_paths:
        zip_path = os.path.join(folder, "框圖jpg_" + stamp + ".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for jpg in jpg_paths:
                z.write(jpg, os.path.basename(jpg))
        paths.append(zip_path)
    b_path = os.path.join(folder, "B_YOLO數量_" + stamp + ".csv")
    write_csv(b_path, B_HEADERS, b_rows)
    paths.append(b_path)
    a_path = os.path.join(folder, "A_三模型判斷_" + stamp + ".csv")
    write_csv(a_path, A_HEADERS, a_rows)
    paths.append(a_path)
    xlsx_path = os.path.join(folder, "辨識結果_" + stamp + ".xlsx")
    if write_excel(xlsx_path, b_rows, a_rows):
        paths.append(xlsx_path)
    return paths


# ---------- 主流程 ----------
def compute_duration(gallery_items, *args):
    """ZeroGPU：依照片張數決定這次要申請幾秒。"""
    count = len(gallery_items) if gallery_items else 1
    return min(BASE_SECONDS + SECONDS_PER_IMAGE * count, MAX_DURATION)


def _run(gallery_items):
    """逐張處理，每處理完一張就更新一次畫面。"""
    conf, iou, imgsz, agnostic, min_area = B_CONFIDENCE, B_IOU, B_IMGSZ, True, B_MIN_AREA
    if not gallery_items:
        yield [], [], [], "請先上傳至少一張照片。", None
        return

    items = list(gallery_items)
    total = len(items)
    names = a_infer.available_models()
    has_b = b_infer.available()

    messages = []
    if len(names) == 0:
        messages.append("A：找不到模型檔，請確認 models 資料夾內有 a_*.pth。")
    if not has_b:
        messages.append("B：找不到模型檔 models/b_yolo.pt，請確認模型檔是否已放入。")

    numeric_b = False
    if has_b:
        try:
            numeric_b = all(str(n).isdigit() for n in b_infer.class_names())
        except Exception:
            numeric_b = False
        if numeric_b:
            messages.append("B 這個權重的類別只有編號、沒有水果名稱，所以 B 表格只列「水果框數」；"
                            "要水果名稱請換上重新訓練的 b_yolo.pt。")

    out_dir = tempfile.mkdtemp()      # 畫好框的 jpg 存在這裡
    used_names = set()
    jpg_paths = []

    a_table, b_table, b_gallery, total_stats = [], [], [], {}

    # 在（ZeroGPU 的）GPU 函式裡才把模型搬到 GPU；沒有 GPU 就留在 CPU
    if len(names) > 0:
        a_infer.use_device()
    if has_b:
        b_infer.use_device()

    try:
        for index, item in enumerate(items, start=1):
            label = "第 " + str(index) + " 張"
            try:
                label, image = load_item(item)

                a_rows = []
                if len(names) > 0:
                    a_rows = analyze_a(image, names)
                    for row in a_rows:
                        a_table.append([label] + row)

                if has_b:
                    b_image, stats = analyze_b(image, conf, iou, imgsz, agnostic, min_area, numeric_b)
                    if len(stats) == 0:
                        # 沒框到：提示並附上 A 的第一名判斷（A 不需要框，沒框到時仍可參考）
                        hint = ""
                        pick = [r for r in a_rows if r and r[0] == "ResNet18"] or a_rows
                        if pick and len(pick[0]) > 1:
                            hint = "A 判斷可能是 " + show_a([[label] + list(pick[0])])[0][2] + "，僅供參考。"
                        messages.append(label + "：沒框到水果（可能是水果太小、太密，或不在 B 認得的類別內）。" + hint)
                    object_count = sum(len(v) for v in stats.values())
                    stem = os.path.splitext(label)[0]
                    jpg_name, k = stem + "_框.jpg", 2
                    while jpg_name in used_names:
                        jpg_name, k = stem + "_框" + str(k) + ".jpg", k + 1
                    used_names.add(jpg_name)
                    jpg_path = os.path.join(out_dir, jpg_name)
                    b_image.save(jpg_path, quality=92)       # 存成 jpg，圖片預覽的下載鍵拿到的就是 jpg
                    jpg_paths.append(jpg_path)
                    b_gallery.append((jpg_path, label + "（" + str(object_count) + " 個）"))
                    if len(stats) == 0:
                        b_table.append([label, "（沒有偵測到水果）", "0", "-"])
                    else:
                        b_table.extend(stats_rows(label, stats))
                        for fruit_name, confs in stats.items():
                            total_stats.setdefault(fruit_name, []).extend(confs)
            except Exception as error:
                messages.append(label + " 處理失敗：" + str(error))

            progress = "處理中：" + str(index) + " / " + str(total)
            yield b_gallery, show_b(b_table), show_a(a_table), "\n\n".join(messages + [progress]), None
    finally:
        if len(names) > 0:
            a_infer.release()
        if has_b:
            b_infer.release()

    if has_b and total > 1 and len(total_stats) > 0:
        b_table.extend(stats_rows("全部合計", total_stats))
    if has_b:
        object_total = sum(len(v) for v in total_stats.values())
        messages.append("B：共處理 " + str(total) + " 張照片，偵測到 " + str(object_total)
                        + " 個物件、" + str(len(total_stats)) + " 種類別。")
    messages.append(NOTE)

    try:
        files = export_files(b_table, a_table, jpg_paths)
    except Exception as error:
        files = None
        messages.append("匯出檔案失敗：" + str(error))

    yield b_gallery, show_b(b_table), show_a(a_table), "\n\n".join(messages), files


def clear_all():
    """清除圖片與所有結果。"""
    return [], [], [], [], None, ""


if HAS_SPACES:
    run = spaces.GPU(duration=compute_duration)(_run)
else:
    run = _run


def supported_text():
    """「本系統認得哪些水果」的說明文字：A 的水果大類、B 的類別名稱。"""
    lines = []
    try:
        loaded = getattr(a_infer, "LOADED", {})
        if loaded:
            families = list(next(iter(loaded.values()))[1])
            lines.append("**A 能分的水果大類（" + str(len(families)) + " 類）**：" + "、".join(bi(f) for f in families))
    except Exception:
        pass
    try:
        if b_infer.available():
            b_names = [str(n) for n in b_infer.class_names()]
            if all(n.isdigit() for n in b_names):
                lines.append("**B 能框的類別**：這個權重只有編號、沒有水果名稱，所以不列出；"
                             "重新訓練的權重會列出名稱。")
            else:
                shown = list(b_names)
                lines.append("**B 能框的水果（" + str(len(shown)) + " 類）**：" + "、".join(bi(n) for n in shown))
    except Exception:
        pass
    lines.append("不在名單內的水果，A 仍會硬分到最接近的一類，B 可能漏框或框成別種；"
                 "Top 1 標「（低信心）」時請特別小心。")
    return "\n\n".join(lines)


# ---------- 網頁介面 ----------
with gr.Blocks(title="水果辨識") as demo:
    gr.Markdown(
        "# 🍎 水果辨識\n"
        "**① 上傳照片（可一次多張）　② 按「辨識」　③ 看右邊結果，需要的話下載。**"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Gallery(label="① 照片（可多張，點一下或拖進來）", type="filepath", columns=3,
                                     height=320, object_fit="contain", interactive=True)
            button = gr.Button("② 辨識", variant="primary", size="lg")
            clear_button = gr.Button("清除圖片與結果")

        with gr.Column(scale=2):
            b_output = gr.Gallery(label="框出的水果（點圖可放大，放大後可下載 jpg）",
                                  columns=2, height=380, object_fit="contain", interactive=False)
            a_table_output = gr.Dataframe(headers=A_HEADERS, datatype="str", interactive=False,
                                          wrap=True, label="這是什麼水果？（A：三個模型各自的判斷）")
            b_table_output = gr.Dataframe(headers=B_HEADERS, datatype="str", interactive=False,
                                          wrap=True, label="框到哪些水果、幾個？（B）")
            download_output = gr.File(label="下載結果（框圖 jpg、CSV、Excel）", file_count="multiple",
                                      interactive=False)
            message_output = gr.Markdown()

    with gr.Accordion("本系統認得哪些水果？（點開查看）", open=False):
        gr.Markdown(supported_text())

    _outputs = [b_output, b_table_output, a_table_output, message_output, download_output]
    button.click(fn=run, inputs=[image_input], outputs=_outputs, api_name="run")
    clear_button.click(fn=clear_all, inputs=None,
                       outputs=[image_input, b_output, a_table_output, b_table_output, download_output, message_output],
                       api_name="clear")


if __name__ == "__main__":
    on_space = os.environ.get("SPACE_ID") is not None
    lan = os.environ.get("LAN") == "1"
    demo.queue()
    # 本機：雙擊 bat 後自動打開瀏覽器；7860 被占用時自動換下一個空的連接埠。
    # Hugging Face（on_space）和區域網路（LAN=1）維持固定 7860，也不自動開瀏覽器。
    fixed = on_space or lan
    demo.launch(
        server_name="0.0.0.0" if fixed else "127.0.0.1",
        server_port=7860 if fixed else None,
        inbrowser=not fixed,
    )
