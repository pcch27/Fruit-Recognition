# 模型權重

## 檔案對照

| 檔案 | 說明 |
|---|---|
| `b_yolo.pt` | **目前使用的 B（26 類）。** 就是 Colab 訓練產生的 `b_yolo_fruit26.pt` 改名而來，Hugging Face Space 放的也是同一個檔案改名成 `b_yolo.pt`。 |
| `a_resnet18.pth`、`a_mobilenetv2.pth`、`a_baseline.pth` | A 分類的三個模型。 |

`b_yolo.pt` 的 SHA-256（可用來確認和 Space 上的是同一個檔案）：

```
a146e225c77c39d52fc66f9adae83ce8cbf75f5846f839dfe612c5f41c001371
```

驗證：Colab 訓練輸出的 `weights/best.pt` 與 `b_yolo.pt` 逐位元相同（MD5 `30cfdee16acef16a634f73de4a716f38`）；A 的三個 `.pth` 也與訓練輸出相同。

## 怎麼來的

1. 用 `Colab/Colab_B.ipynb`（專案根目錄） 訓練：YOLOv8s、50 epochs、batch 16、640 px，資料設定是 `B/data_fruit.yaml`（26 類）。
2. 訓練完 notebook 把 `best.pt` 另存成 `b_yolo_fruit26.pt`。
3. 要放進網頁時，把 `b_yolo_fruit26.pt` **改名為 `b_yolo.pt`**（`app.py` 讀的檔名固定是 `b_yolo.pt`）。

## 授權

權重由下列資料訓練而來，**以 CC BY-SA 4.0 釋出**（Creative Commons Attribution-ShareAlike 4.0）：使用或改作時請標註出處，並以相同條款分享。程式碼的授權是 MIT（見根目錄 `LICENSE`），兩者不同。

訓練資料出處：

- Fruits-360（Mihai Oltean，Kaggle），CC BY-SA 4.0；原始論文 Horea Mureşan、Mihai Oltean，2018。
- Multi-Class Fruit and Vegetable Detection Dataset（George Bayón, Mencía；Sánchez González, Lidia（Editor）；Universidad de León，2026，Zenodo 10.5281/zenodo.18904638），CC BY 4.0。
- LVIS Fruits And Vegetables Dataset（Henning Heyen；原始資料集 LVIS：Gupta、Dollár、Girshick，2019；影像來源 COCO；GitHub：<https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset>）：LVIS 標註 CC BY 4.0；影像依 COCO 相關授權條款使用。
- FruitDet（Fruit Dataset Annotation，sirunchained，Hugging Face：<https://huggingface.co/datasets/sirunchained/fruit-dataset-annotation>），CC BY 4.0。
- Jackfruit Dataset（VietNam Fruit in Lab，Roboflow Universe，2023：<https://universe.roboflow.com/vietnam-fruit-in-lab/jackfruit-maj1o>），CC BY 4.0。
- Cantaloupe Dataset（lamhuy1810，Roboflow Universe：<https://universe.roboflow.com/lamhuy1810-8dest/cantaloupe-idfms>），CC BY 4.0。

B 的 YOLOv8s 使用 Ultralytics 套件（AGPL-3.0）訓練；若要把整個系統當服務對外提供，請另外確認 AGPL 的要求。
