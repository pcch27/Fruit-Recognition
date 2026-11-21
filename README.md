# 水果影像辨識系統 Fruit Recognition

整合水果影像分類與物件偵測，包含 A 分類（Baseline CNN、ResNet18、MobileNetV2）及 B 偵測（YOLOv8s，26 類），並使用 Gradio 建立操作介面。

**線上 Demo：** [Hugging Face Spaces – Fruit Recognition](https://huggingface.co/spaces/PCChen27/Fruit-Recognition)  
**專題簡報：** `docs/水果影像辨識_作品集.pdf`、`docs/水果影像辨識_完整版.pdf`

## 重點結果（取自專題簡報）

- **A 分類（Top-1），** ResNet18 在白底照片與真實照片的準確率分別為 98.5% 與 66.0%；MobileNetV2 分別為 95.0% 與 64.9%；Baseline CNN 則為 85.4% 與 9.3%。
- **加入真實照片訓練的改善，** ResNet18 在真實照片上的 Top-1 Accuracy 由 15.5% 提升至 66.0%；MobileNetV2 則由 19.6% 提升至 64.9%。
- **B 偵測（26 類，資料集測試集），** mAP@0.5 為 0.772、mAP@0.5:0.95 為 0.501，Precision 為 74.0%、Recall 為 75.9%。
- **真實與圖庫照片測試（175 張，B 的 26 類範圍），** 共 121 張照片框到至少一個正確水果種類，命中率為 69.1%。其中 real_test109 為 72／101 張，stock_test_set 為 49／74 張。
- **A、B 整合測試（157 張單一水果照片），** 在 A、B 皆支援的水果範圍內，至少有一個模型答對的比例為 79.6%。

B 偵測在 175 張真實與圖庫照片上的逐張測試結果，可透過 `Local/results/B_逐張清單_175張.xlsx` 重新計算，與專題簡報一致。

A 分類的 Top-1 Accuracy 與 B 偵測的 mAP 數據取自 Google Colab 訓練輸出，若要重新計算，需使用原始資料集及測試影像。

## 操作方式

介面只有「上傳照片」和「辨識」，使用內建偵測參數，不需要手動調整。

- 預設信心閾值 0.35、重疊框處理閾值 0.5、圖片尺寸 640 px。
- 完全沒有偵測到水果時，會提示並附上 A 的第一名判斷供參考。

啟動方式（在 `Local/` 資料夾）：

- 執行 `python app.py`。
- Windows 可雙擊 `start_app.bat`。

## 本機執行

需要 Python 3.10～3.12，模型權重已放在 `Local/models/`。以下指令均在 `Local/` 資料夾內執行。

安裝套件：

```bash
pip install -r requirements.txt
```

啟動：

```bash
python app.py
```

Windows 也可以使用批次檔：

- `run_local.bat`：第一次執行時建立 `.venv` 環境並啟動。
- `start_app.bat`：啟動介面。

啟動後，在瀏覽器開啟：

[http://127.0.0.1:7860](http://127.0.0.1:7860)

## 專案結構

```text
Fruit-Recognition/
├── README.md                      專案說明
├── FILE_STRUCTURE.md              完整檔案結構說明
├── LICENSE                        程式碼授權（MIT）
├── .gitignore
│
├── Colab/                         Google Colab 模型訓練
│   ├── Colab_A.ipynb              A 分類：Baseline CNN、ResNet18、MobileNetV2
│   ├── colab_A_stage1_check.ipynb A 階段 1（只用白底圖）檢查
│   └── Colab_B.ipynb              B 偵測：YOLOv8s，26 類
│
├── Local/                         本機執行與測試
│   ├── app.py                     Gradio 介面
│   ├── a_infer.py                 A 分類推論
│   ├── a_common.py                A 分類共用函式
│   ├── a_data_utils.py            A 分類資料處理
│   ├── b_infer.py                 B 偵測推論
│   ├── requirements.txt           本機需要的套件
│   ├── run_local.bat              第一次執行：建立環境並啟動
│   ├── start_app.bat              啟動介面
│   ├── models/                    模型權重
│   │   ├── a_resnet18.pth
│   │   ├── a_mobilenetv2.pth
│   │   ├── a_baseline.pth
│   │   └── b_yolo.pt
│   ├── examples/                  介面範例照片（Fruits-360）與出處
│   ├── A/                         分類資料整理、訓練及評估程式
│   ├── B/                         偵測資料合併、訓練、評估程式、類別名單及資料設定
│   ├── experiments/               175 張測試照片答案、74 張圖庫照片出處及比對工具
│   ├── results/                   B 逐張清單（175 張）、A 階段 1 與最終結果、A 與 B（26 類）訓練輸出
│   ├── tools/                     compare_a.py、dup_check.py、resize_images.py
│   └── tests/                     不需 GPU 的測試（python tests/test_all.py）
│
├── HuggingFace/                   Hugging Face Spaces 部署檔案
│   ├── README.md                  Space 說明頁
│   ├── requirements.txt           Space 需要的套件
│   ├── packages.txt               Space 需要的系統套件
│   ├── samples/                   範例照片，出處見 README.txt
│   └── UPLOAD.md                  上傳檔案清單
│
└── docs/                          簡報：作品集、完整版（PDF）
```

## 模型訓練與部署

使用 Google Colab 搭配 T4 GPU 進行模型訓練。

1. 將專案 ZIP 與 Fruits-360 的 `archive.zip` 放入 Google Drive。
2. 執行 `Colab/Colab_A.ipynb`，訓練 A 分類模型。
3. 執行 `Colab/Colab_B.ipynb`，訓練 B YOLOv8s 偵測模型。
4. 將訓練完成的模型權重放入 `Local/models/`。
5. Hugging Face Spaces 所需檔案及上傳方式，參考 `HuggingFace/UPLOAD.md`。

## 授權與資料來源

### 程式碼與模型權重

- **程式碼：** MIT License，詳見 `LICENSE`。
- **模型權重（`Local/models/`）：** CC BY-SA 4.0，相關授權說明見 `Local/models/README.md`。
- **範例照片（`Local/examples/`）：** Fruits-360，CC BY-SA 4.0，來源及出處見 `Local/examples/README.md`。
- **簡報內嵌的第三方照片：** 版權屬原攝影師，不包含在上述授權範圍內。

### 訓練資料來源

**Fruits-360**

- 作者：Mihai Oltean。
- Kaggle：[Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)。
- 授權：CC BY-SA 4.0，使用時需標註出處，改作需以相同條款分享。
- `real_test109` 取自 multi 資料夾（Version 2026.09.20.0）的多水果測試影像，本專案從中整理 109 張未參與 A 模型訓練的照片作為測試資料，正確答案由本專案自行標註。

原始論文：

Horea Mureşan、Mihai Oltean，*Fruit recognition from images using deep learning*，Acta Univ. Sapientiae, Informatica 10(1), 26–42, 2018。

**Multi-Class Fruit and Vegetable Detection Dataset**

- 作者：George Bayón, Mencía。
- 編輯：Sánchez González, Lidia。
- 機構：Universidad de León。
- 年份：2026。
- Zenodo：[10.5281/zenodo.18904638](https://doi.org/10.5281/zenodo.18904638)。
- 授權：CC BY 4.0。

**其他補充資料**

- LVIS。
- FruitDet（Hugging Face）。
- Jackfruit（Roboflow）。
- Cantaloupe（Roboflow）。

### 圖庫測試照片

74 張圖庫測試照片取自 Unsplash 及 Pexels，攝影師與原始頁面連結整理於 `Local/experiments/stock_test_set.csv`。

簡報中使用的圖庫照片包括：

- Unsplash：LuAnn Hunt（Peach_01）、Mekht（Pear_01），使用於作品集第 5 頁及完整版第 20 頁。
- Pexels：Gustavo Denuncio（Kiwi_01）、Terrance Barksdale（Cantaloupe_02）。

原始圖庫照片沒有放入本專案儲存庫，請勿將原始照片整包重新散布。

## 限制與注意事項

- B 的測試集僅包含 B 可辨識的 26 類水果。油桃、龍眼及扁油桃相關照片共 14 張已移除（real_test109 8 張，含 1 張含油桃的多種水果照片；stock_test_set 6 張），剩餘 175 張。A 的測試集另行計算，A 可辨識的水果（例如油桃）仍保留在 A 的評估中。這些測試照片數量有限且經過人工挑選，不能代表所有實際使用情況。B 在這 175 張照片中，有 121 張框到正確種類，命中率為 69.1%。
- 水果名稱是否符合答案，由程式依模型辨識輸出判定，實際使用前建議抽查模型輸出結果。
- B 的框選結果是模型預測，並非人工標註。統計數量是預測框數，不等於實際水果數量。水果重疊或被遮住時（例如堆疊的哈密瓜），可能出現重複計算或漏檢。
- 程式計算的「正確框」可能高估實際辨識表現。例如預測框的名稱出現在該照片的答案中，即使框選位置實際落在另一種水果上，也可能被計算為正確。多種水果同時出現的照片尤其需要注意。
