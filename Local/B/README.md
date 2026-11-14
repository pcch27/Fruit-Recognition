# B 偵測（YOLOv8s）

| 檔案 | 說明 |
|---|---|
| `data_fruit.yaml` | 26 類的資料集設定。`path: /content/dfv_fruit` 是 Colab 上的路徑，本機重跑要改成自己的資料夾。 |
| `b_fruit_classes.txt` | 21 類水果名單（舊版，已拿掉 9 類蔬菜）。 |
| `b_fruit26_classes.txt` | 26 類名單 = 21 類 + Kiwi、Raspberry、Blackberry、Cantaloupe、Jackfruit。順序固定，編號 21～25 是新增類別。 |
| `B0i_merge_curated.py` | 把補訓資料合併進 21 類資料集變成 26 類：先確認前 21 類的名稱與順序完全相同，新增類別的訓練圖片太少時會重複放入。 |

## 訓練流程

1. 21 類水果資料集（拿掉 9 類蔬菜）。
2. 用 `B0i_merge_curated.py` 合併補訓資料，變成 26 類。
3. 訓練：`Colab/Colab_B.ipynb`（專案根目錄）（50 epochs、batch 16、640 px）。
4. 權重存成 `b_yolo_fruit26.pt`，放進網頁時改名為 `models/b_yolo.pt`。

## 沒有放進這個儲存庫的東西

- **補訓資料** `fruit26_extra_dataset_curated.zip`（約 639 張圖片，來自 LVIS、Jackfruit、cantaloupe 等）：檔案大，而且各來源授權不同，公開前要逐一確認。
- **`stock_test_set`、`real_test109` 的測試照片不可放進訓練資料**，它們只用來比較新舊模型。
