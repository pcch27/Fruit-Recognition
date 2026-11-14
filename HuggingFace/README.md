---
title: Fruit Recognition
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 水果影像辨識系統 Fruit Recognition

整合影像分類（Image Classification）與物件偵測（Object Detection）技術，運用深度學習模型開發水果影像辨識系統。

本系統為深度學習實作成果，辨識範圍以模型訓練時涵蓋的水果類別為主，預測分數僅供模型判斷參考，不等同於實際辨識準確率。

系統提供兩種辨識功能：

- **A｜水果影像分類：** 使用 MobileNetV2、ResNet18、Baseline CNN 三種模型，分別預測照片中最可能的水果類別，共 55 類。
- **B｜水果物件偵測：** 使用 YOLOv8s 偵測照片中的水果，標示水果位置、種類並統計數量，共 26 類。

支援一次上傳多張照片，並提供辨識結果、標註圖片及 CSV、Excel 檔案下載。

## 一、辨識類別

### A｜影像分類（55 類）

可辨識的水果大類包括：

apple（蘋果）、apricot（杏桃）、avocado（酪梨）、banana（香蕉）、blackberry（黑莓）、blueberry（藍莓）、cactus（仙人掌果）、cantaloupe（哈密瓜）、carambola（楊桃）、cherimoya（番荔枝）、cherry（櫻桃）、clementine（小柑橘）、cocos（椰子）、dates（椰棗）、fig（無花果）、gooseberry（醋栗）、granadilla（甜百香果）、grape（葡萄）、grapefruit（葡萄柚）、guava（芭樂）、huckleberry（越橘）、kaki（柿子）、kiwi（奇異果）、kumquats（金桔）、lemon（檸檬）、limes（萊姆）、lychee（荔枝）、mandarine（柑橘）、mango（芒果）、mangostan（山竹）、maracuja（百香果）、melon（甜瓜）、mulberry（桑葚）、nectarine（油桃）、orange（橘子）、papaya（木瓜）、passion（百香果）、peach（桃子）、pear（梨子）、pepino（香瓜茄）、physalis（酸漿果）、pineapple（鳳梨）、pitahaya（火龍果）、plum（李子）、pomegranate（石榴）、pomelo（柚子）、quince（榲桲）、rambutan（紅毛丹）、raspberry（覆盆子）、redcurrant（紅醋栗）、salak（蛇皮果）、strawberry（草莓）、tamarillo（樹番茄）、tangelo（橘柚）、watermelon（西瓜）。

### B｜物件偵測（26 類）

可偵測並框選的水果類別包括：

Apple（蘋果）、Apricot（杏桃）、Banana（香蕉）、Cherry（櫻桃）、DragonFruit（火龍果）、Durian（榴槤）、Fig（無花果）、Grape（葡萄）、Lemon（檸檬）、Lime（萊姆）、Lychee（荔枝）、Mango（芒果）、Orange（橘子）、Peach（桃子）、Pear（梨子）、Pineapple（鳳梨）、Plum（李子）、Pomegranate（石榴）、Quince（榲桲）、Strawberry（草莓）、Watermelon（西瓜）、Kiwi（奇異果）、Raspberry（覆盆子）、Blackberry（黑莓）、Cantaloupe（哈密瓜）、Jackfruit（菠蘿蜜）。

**辨識限制：** A 僅能從 55 個既有類別中選擇預測結果，名單外的水果仍可能被分類為其他類別；B 僅能偵測 26 個指定類別，名單外的水果可能漏檢或誤判。A 若出現「（低信心）」標示，表示模型對預測結果的信心不足。

## 二、使用方式

選擇影像分類或物件偵測功能，上傳一張或多張水果照片，按下「辨識」即可執行模型預測，查看水果分類結果或物件偵測標註圖片，並支援 CSV、Excel 格式的辨識結果下載。

專案另提供 `samples/` 資料夾，內含 6 張範例照片，可供使用者上傳測試。

## 三、訓練資料來源

### A｜影像分類

**資料集：Fruits-360**

作者／發布者：Mihai Oltean

原始研究：Mureşan, H.、Oltean, M.（2018）

資料來源：[Kaggle - Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)

授權：CC BY-SA 4.0

### B｜物件偵測

**主要訓練資料**

**Multi-Class Fruit and Vegetable Detection Dataset for Computer Vision**

作者／發布者：George Bayón, Mencía；Sánchez González, Lidia（Editor），Universidad de León

資料來源：[Zenodo](https://zenodo.org/records/18904638)

授權：CC BY 4.0

**補充訓練資料**

**1. LVIS Fruits And Vegetables Dataset**

作者／發布者：Henning Heyen

原始資料集：LVIS（Gupta, A.、Dollár, P.、Girshick, R., 2019）

影像來源：COCO

資料來源：[GitHub - Fruits-And-Vegetables-Detection-Dataset](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset)

授權：LVIS 標註 CC BY 4.0；影像依 COCO 相關授權條款使用。

**2. FruitDet（Fruit Dataset Annotation）**

作者／發布者：sirunchained

資料來源：[Hugging Face - Fruit Dataset Annotation](https://huggingface.co/datasets/sirunchained/fruit-dataset-annotation)

授權：CC BY 4.0

**3. Jackfruit Dataset**

作者／發布者：VietNam Fruit in Lab

平台：Roboflow Universe（2023）

資料來源：[Roboflow - Jackfruit Dataset](https://universe.roboflow.com/vietnam-fruit-in-lab/jackfruit-maj1o)

授權：CC BY 4.0

**4. Cantaloupe Dataset**

作者／發布者：lamhuy1810

平台：Roboflow Universe

資料來源：[Roboflow - Cantaloupe Dataset](https://universe.roboflow.com/lamhuy1810-8dest/cantaloupe-idfms)

授權：CC BY 4.0

## 四、外部測試資料（未參與訓練）

為觀察模型在不同影像條件下的辨識表現，另外整理外部測試資料，未納入模型訓練。

`real_test109` 取自 Fruits-360 多水果（Multi-Fruit）測試資料，原整理共 109 張；因 B 模型不支援油桃，排除 7 張油桃照片及 1 張含油桃的多水果照片，最終使用 101 張進行外部測試。

`stock_test_set` 則收集自 Unsplash、Pexels 的水果照片，共 74 張，並涵蓋 26 類範圍內的水果。

外部測試資料合計 175 張，其中 `stock_test_set.csv` 記錄照片的攝影師及原始圖片連結。

## 五、範例照片來源（samples/）

專案 `samples/` 資料夾提供 6 張範例照片，其中 apple_32.jpg、apple_orange_2.jpg、cherry_44.jpg、pear_10.jpg 取自 Fruits-360 資料集（Mihai Oltean），資料來源：[Kaggle - Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)。

另外，Kiwi_01.jpg 由 Gustavo Denuncio 拍攝，取自 [Pexels](https://www.pexels.com/photo/fresh-kiwi-fruits-displayed-in-trays-31558795/)；Cantaloupe_02.jpg 由 Terrance Barksdale 拍攝，取自 [Pexels](https://www.pexels.com/photo/pile-of-melons-18979310/)。

## 六、授權資訊

本專案程式碼採用 MIT License，模型權重採用 CC BY-SA 4.0 授權。

訓練資料方面，Fruits-360 採用 CC BY-SA 4.0 授權；Zenodo、FruitDet、Roboflow Jackfruit Dataset 及 Roboflow Cantaloupe Dataset 均採用 CC BY 4.0 授權。LVIS 標註採用 CC BY 4.0 授權，影像則依 COCO 相關授權條款使用。

外部測試照片方面，Unsplash 圖片依 Unsplash License 使用，Pexels 圖片依 Pexels License 使用。

所有資料集、模型權重及外部圖片均依各自授權條款使用，並保留原始來源資訊。

## 七、專案連結

GitHub：[Fruit Recognition](https://github.com/pcch27/Fruit-Recognition)
