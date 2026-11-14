# 上傳到 Hugging Face Space 的檔案

Space 的根目錄要有下面這些檔案（從本專案複製過去）：

| 從哪裡複製 | 檔案 | 說明 |
|---|---|---|
| `HuggingFace/` | `README.md` | Space 說明頁（最上面的設定區塊不要刪） |
| `HuggingFace/` | `requirements.txt`、`packages.txt` | Space 要安裝的套件 |
| `HuggingFace/` | `samples/`（放範例照片） | 供使用者下載測試 |
| `Local/` | `app.py` | Gradio 介面 |
| `Local/` | `a_common.py`、`a_data_utils.py`、`a_infer.py`、`b_infer.py` | A、B 推論程式 |
| `Local/models/` | `a_resnet18.pth`、`a_mobilenetv2.pth`、`a_baseline.pth`、`b_yolo.pt` | 模型權重，放在 Space 的 `models/` 資料夾 |

不需要上傳：`Local/` 裡的 `A/`、`B/`、`experiments/`、`results/`、`tests/`、`tools/`、`examples/`、`diagnose_b.py` 與 `.bat` 檔，以及 `Colab/`、`docs/`。

Space 預設跑一般模式（不設定 `RESEARCH`）。
