# -*- coding: utf-8 -*-
# +
#!/bin/bash
file="inference_mrxs_list.txt"
username="u2204495"

while IFS="." read -r case_name ext; 
do

    echo "================================================"
    echo "Now Inference $case_name"
    echo "================================================"
    
#     echo "\n--------------- 1. Generate WSI pkl file ---------------"
    
#     # 生成 pkl 檔案給 Inference.py 用
#     python immune_inf/preprocess.py --wsi_path "/work/$username/liver_portal/wsi/train/$case_name.$ext" \
#                          --pkl_path "/work/$username/liver_portal/wsi/test/inference_result/RRRRRR/pkl/$case_name.pkl" \
#                          --segment_output_dir "/work/$username/liver_portal/wsi/test/inference_result/RRRRRR/tissue_png_imgs" \
#                          --patch_saving_dir "/work/$username/liver_portal/wsi/test/inference_result/RRRRRR/patch"

#     echo "\n--------------- 2. Inference Portal Area ---------------"

#     # inference Portal 區域，生成 WSI 大小的 mask
#     python inf_portal+tissue.py --wsi_path "/work/$username/liver_portal/wsi/train/$case_name.$ext" \
#         --patch_size 512 \
#         --patch_overlap_ratio 0.25 \
#         --batch_size 256 \
#         --output_type "polygon" \
#         --model_dir "/work/$username/liver_portal/linxuan_model/1031_fine_zung/Unet_10312037_83.07_best.ckpt" \
#         --immune_model_dir "/work/$username/Immune_Cell/train/weights/best_model_78.54.pt" \
#         --output_dir "/work/$username/liver_portal/wsi/test/inference_result"

#     echo "\n--------------- 3. Inference Immune Cell ---------------"

#     # inference 免疫細胞，生成 WSI 大小的 mask
#     python immune_inf/Inference.py --wsi_path "/work/$username/liver_portal/wsi/train/$case_name.$ext" \
#                         --pkl_path "/work/$username/liver_portal/wsi/test/inference_result/RRRRRR/pkl/$case_name.pkl" \
#                         --inference_dir "/work/$username/liver_portal/wsi/test/inference_result/RRRRRR/result" \
#                         --immune_cell_weights "/work/$username/Immune_Cell/train/weights/best_model_90.77.pt" \
#                         --num_class 2 \
#                         --batch_size 8 \
#                         --cell_model "smp" \
#                         --save_mask True \
#                         --save_all_cells True \
#                         --filter_size 8 

    echo "\n--------------- 4. Calculate clustered immune cells ---------------"
    
    # 生成熱成像圖 和 filter 後的免疫細胞標註圖
    python calculate_clust_immune.py --immune_map "/work/$username/liver_portal/wsi/test/inference_result/RRRRRR/result/MASK/${case_name}_whole_mask.png" \
        --portal_map "/work/$username/liver_portal/wsi/test/inference_result/${case_name}_portal_binary.png" \
        --wsi_path "/work/$username/liver_portal/wsi/train/$case_name.$ext" \
        --output_path "/work/$username/liver_portal/wsi/test/inference_result"

done < "$file"
