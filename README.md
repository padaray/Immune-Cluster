# Immune-Cluster
本專案用於分析肝臟病理影像，包含 Portal 區域與免疫細胞的偵測、分析與熱圖生成。

</br>

## ⚙️ 環境安裝
當中使用的 smp.MANet 是使用[這篇連結](https://github.com/qubvel-org/segmentation_models.pytorch):  
```
# 安裝指令
$ pip install git+https://github.com/qubvel/segmentation_models.pytorch
```
其他環境安裝則使用 **environment.yml**

</br>

## 📂 目錄結構
```
├── 🖥️ run_inference.sh # 自動執行完整分析流程
├── 📄 inf_portal+tissue.py # inference Portal 區域
├── 📄 calculate_clust_immune.py # 免疫細胞群聚分析與熱圖生成
├── 📑 inference_mrxs_list.txt # 需進行推論的影像列表
├── 📑 environment.yml # 環境檔
└── 📂 immune_inf # 亦豪的 inference code，用來 inf 亦豪的 weight
     ├── 📄 preprocess.py # WSI 切割 Patch 後(過濾非肝區域)，轉成 pkl 檔
     ├── 📄 Inference.py # inference 免疫細胞\
```

</br>

## 🖥️ 使用方式 (執行 run_inference.sh)

執行 run_inference.sh 後，會依序執行以下 python 檔

#### 1. preprocess.py
功能：切割 Whole Slide Image (WSI) 成 patch，並去除非肝臟區域，最後轉成 pkl 檔
| 參數名稱  | 說明 |
| ------------- | ------------- |
| wsi_path  | Inference 的 WSI 路徑 |
| pkl_path  | pkl 檔案儲存路徑  |
| segment_output_dir  | 組織區域影像儲存路徑 |
| patch_saving_dir  | 切好後的 patch 儲存路徑  |

#### 2. inf_portal+tissue.py
功能：Inference Portal 區域，印出 WSI 大小的 Mask 圖(20x)。
| 參數名稱  | 說明 |
| ------------- | ------------- |
| wsi_path  | Inference 的 WSI 路徑 |
| model_dir  | Portal 模型路徑 |
| output_dir  | 印出的 Mask 儲存路徑 |


#### 3. Inference.py
功能：Inference Immune，印出 WSI 大小的 Mask 圖(20x)。
| 參數名稱  | 說明 |
| ------------- | ------------- |
| wsi_path  | Inference 的 WSI 路徑 |
| pkl_path  | 之前 pkl 檔案儲存路徑  |
| inference_dir | 印出的 Mask 儲存路徑 |
| immune_cell_weights  | Immune Cell 模型路徑 |


#### 4. calculate_clust_immune.py
功能：計算免疫細胞群聚情形，並生成熱成像圖與標註影像。
| 參數名稱  | 說明 |
| ------------- | ------------- |
| immune_map  | Immune Cell Mask 儲存路徑 |
| portal_map   | portal 區域 Mask 儲存路徑 |
| wsi_path   | Inference 的 WSI 路徑 |
| output_path   | 熱成像圖的儲存路徑 |
  
