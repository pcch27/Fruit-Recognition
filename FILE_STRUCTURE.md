# 檔案結構

```
Fruit-Recognition/
├── README.md                      專案說明（中英）
├── FILE_STRUCTURE.md              本檔案結構圖
├── LICENSE                        程式碼授權（MIT）
├── .gitignore
│
├── Colab/                         ① Google Colab 訓練（上傳到 Google Drive，用 T4 GPU 執行）
│   ├── Colab_A.ipynb              A 分類：Baseline CNN、ResNet18、MobileNetV2
│   ├── colab_A_stage1_check.ipynb A 階段 1（只用白底圖）檢查
│   └── Colab_B.ipynb              B 偵測：YOLOv8s，26 類
│
├── Local/                         ② 本機執行與測試（下載後在這個資料夾裡操作）
│   ├── app.py                     Gradio 介面
│   ├── a_infer.py  a_common.py  a_data_utils.py   A 分類：推論、共用函式、資料處理
│   ├── b_infer.py                                 B 偵測：推論
│   ├── requirements.txt           本機需要的套件
│   ├── run_local.bat              第一次執行：建立環境並啟動
│   ├── start_app.bat           啟動介面
│   ├── models/                    模型權重（a_resnet18.pth、a_mobilenetv2.pth、a_baseline.pth、b_yolo.pt）
│   ├── examples/                  介面範例照片（Fruits-360）與出處
│   ├── A/                         A 分類：整理資料、訓練、評估程式
│   ├── B/                         B 偵測：資料合併、訓練、評估程式、類別名單、資料設定
│   ├── experiments/               175 張測試照片的答案、74 張圖庫照片出處（stock_test_set.csv）、比對工具
│   ├── results/                   結果表：B 逐張清單（175 張）、A 階段 1 vs 最終、A 與 B（26 類）訓練輸出
│   ├── tools/                     小工具：compare_a.py、dup_check.py、resize_images.py
│   └── tests/                     不需 GPU 的測試：python tests/test_all.py
│
├── HuggingFace/                   ③ Hugging Face Space 上傳用（上傳清單見 UPLOAD.md）
│   ├── README.md                  Space 的說明頁（上傳到 Space 根目錄，檔名維持 README.md）
│   ├── requirements.txt           Space 需要的套件
│   ├── packages.txt               Space 需要的系統套件
│   ├── samples/                   範例照片（照片放這裡，出處見 README.txt）
│   └── UPLOAD.md                  要上傳哪些檔案
│
└── docs/                          簡報：作品集、完整版（pptx）與作品集 PDF
```
