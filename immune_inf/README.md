# File & Directory Structure

$1 == username in TWCC
$2 == inference version
$case_name == file name of WSI
$ext == extension name of WSI

--$1/home/
    |--Immune_cell_test/
        |--utils/
            |--remove_dirt.py
            |--tools.py
            |--unet_model.py
            |--unet_model.py
            |--unet_seg_new_weight_221122.pth
        |--utils2/
            |--__init__.py
            |--callbacks.py
            |--dataloader_medical.py
            |--dataloader.py
            |--utils_fit.py
            |--utils_metrics.py
            |--utils.py
        |--nets
            |--__init__.py
            |--resnet.py
            |--unet_training.py
            |--unet.py
            |--vgg.py
        |--models/
            |--atten_unet.py
            |--modules.py
            |--res_unet_plus.py
            |--res_unet.py
            |--unet.py
        |--cluster_selection.py
        |--committee_gt_inf.py
        |--dataset.py
        |--difficulty_measurer.py
        |--evaluate_case.txt
        |--evaluate_patch.py
        |--evaluate_patch.sh
        |--Inference.py
        |--json2xml.py
        |--postprocess_functions.py
        |--preprocess_patch.py
        |--preprocess_tif.py
        |--preprocess.py
        |--result_csv.py
        |--run_committee_gt.py
        |--run_difficulty.py
        |--run.sh
        |--show_pkl.py
        |--test_case_list.py
        |--tif2svs.py
        |--timer.py

--$1/work/
    |--Immune_Cell/
        |--patch/
            |--$case_name/
                |--$case_name_x_y.png
                .
                .
                .
        |--pkl/
            |--$case_name.pkl
            .
            .
            .
        |--result/
            |--$2/
                |--All_Cells/ (patch image with cell contours )
                    |--$case_name/
                        |--$case_name_x_y.png
                        .
                        .
                        .
                |--Cells_with_Portal/ (patch image with cell and portal contours)
                    |--$case_name/
                        |--$case_name_x_y.png
                        .
                        .
                        .
                |--Cells_wo_Portal/ (patch image with cell contours not )
                    |--$case_name/
                        |--$case_name_x_y.png
                        .
                        .
                        .
                |--JSON/
                    |--$case_name/
                |--TXT/
                    |--$case_name/
                |--XML/
                    |--$case_name/
            |--ground_truth/
                |--gt.json
    |--tissue_png_imgs/
            |--$case_name_mask.png
            |--$case_name_output_mask.png
            |--$case_name_output.png
            |--$case_name_rm.png
        |--weights/
            |--Immune_Atten_Unet/
                |--v1_3_1
                    |--best_model.pt
            |--Immune_Cell_ResUnet_PlusPlus/
                |--v2_0_1
                    |--best_model.pt
            |--Portal_Atten_Unet/
                |--v1_0_220922/
                    |--best_model.ckpt
        |--wsi/
            |--$case_name.$ext

# Into. to each .py File

## Main Execution File

0. run.sh
   0-1 吃 test_case_list.txt 的 casename 跟 ext 執行
   0-2 記得修改 username
   0-3 執行須輸入版本號
   0-4 版本號規則：model_version.weight_version.post_process_version
1. preprocess.py
   1-1 使用君將模型移除WSI上馬克筆跟髒污，產生組織區域mask
   1-2 依mask在 指定倍率 下切patch存xy座標成$casename.pkl檔
   1-3 模型偵測後之png影像存於segment_output_dir
   1-4 patch_saving_dir若不為None，另存切出原圖
   1-5 預設patch倍率為40倍
   1-6 另有remove_non_liver暴力移除非肝臟組織，有例外情形存在
2. show_pkl.py
   2-1 印出$casename.pkl第一筆資料供確認
3. Inference.py
   3-1 dataset getitem依$casename.pkl內儲存之座標在指定倍率下切patch（default倍率40、patch size (512, 512) ）
   3-2 進行inference後，經過watershed處理，產生immune cell 點和contour的json檔（座標為指定倍率下之絕對座標，檔名為raw_annotation.json）
   3-3 portal_weights 若不為 None，會import portal偵測的model和weight
   3-4 cell_model 選擇 "AttUnet" 或 "RUnet++"，注意與weight搭配
4. cluster_selection.py
   4-1 讀取raw_annotation.json針對WSI做DBSCAN cluster處理，排除非cluster immune cells
   4-2 點存成result_point_annotation.json、result_point_annotation.xml
   4-3 contour 存成result_contour_annotation.json、result_contour_annotation.xml

## Functional File

1. timer.py

   - 計算function執行時間的decoractor
2. postprocess_functions.py

   - 各後處理function：water、contour產生點座標、移除portal中cell
3. json2xml.py

   - 將json格式轉為xml格式

## Difficulty measurer

1. run_committee_gt.sh

   - 評估資料經 leica & 3D model 後的預測結果
   - 儲存預測結果
2. run_difficulty.sh

   - 使用預測結果計算 difficulty 並存成 json

## Real classifier
