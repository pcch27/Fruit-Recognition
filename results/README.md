# results：實驗結果表（只放表格，不放圖片）

| 檔案 | 內容 |
|---|---|
| `B_逐張清單_175張.xlsx` | B（26 類）在 175 張測試照片（real_test109 101 張 + stock_test_set 74 張；B 認不得的 14 張已移除）的逐張輸出、框數、對錯，與彙總（有框到比例 69.1%、總框 2,249、種類正確的框 1,245）。簡報中 B 在 175 張外部照片的數字由這張表算出。 |
| `A_stage1_vs_final.csv` | A 分類：只用白底圖（階段 1）vs 加入真實照片後（最終）的 Top-1／Top-3，97 張真實單一水果照片。 |

注意：

- 「種類對／錯」由程式依正確答案計算；多種水果的照片會高估（見根目錄 README 的「限制」）。
- B 與 A 的測試集各自計算：B 只放 B 認得的 26 類（所以沒有油桃、龍眼、扁油桃）；A 的評估用 `a/*_eval.json`（97 張單一水果＋12 張多種水果，含 A 認得的油桃）。
- 框圖沒有放進來：圖庫照片加框後仍是衍生物，不能散布。

## 訓練輸出（Colab）

只放小檔：指標、曲線、混淆矩陣。**沒放**權重（已在 `models/`）、`train_batch*.jpg`／`val_batch*.jpg`（內容是資料集圖片）、訓練用的真實照片。

| 資料夾 | 內容 |
|---|---|
| `a/` | A 三個模型：`*_eval.json`（白底 test 與真實照片 Top-1／Top-3，97 張單一水果＋12 張多種水果）、`*_metrics.json`、`*_history.json`、`*_per_family.csv`、混淆矩陣與訓練曲線 |
| `b_fruit26/` | **B（26 類）** 的 YOLO 訓練輸出：`results.csv`（50 epochs）、`results.png`、混淆矩陣、PR／F1／P／R 曲線、`args.yaml` |

B 的 test 數字（P 0.740、R 0.759、mAP@0.5 0.772、mAP@0.5:0.95 0.501）在 `Colab/Colab_B.ipynb`（專案根目錄） 的輸出裡；A 的 Top-1 在 `a/*_eval.json` 與 `A_stage1_vs_final.csv`。
