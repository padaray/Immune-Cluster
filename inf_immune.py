# region
import os
import math
import argparse
import torch
import numpy as np
import cv2
import pyvips as vi
import torch.utils.data as data
from tqdm import tqdm
import segmentation_models_pytorch as smp
from liver_utils import slide_ds as SlideDataset
from torchvision import transforms

# 1. 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 載入模型函數
def load_model(weight_path, num_classes):
    model = smp.MAnet(
        encoder_name="resnet50",       # encoder 名稱
        encoder_weights=None,         # 推論時不需要預訓練權重
        in_channels=3,                # RGB 圖像
        classes=num_classes           # 分類數
    )
    # 載入權重
    checkpoint = torch.load(weight_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # 如果權重直接是 state_dict
    model.to(device)
    model.eval()
    return model

# 3. 預處理函數
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if image is None:
        raise FileNotFoundError(f"圖像檔案不存在或無法讀取: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 轉為 RGB
    image = cv2.resize(image, (512, 512))  # 確保尺寸符合模型要求
    image = transform(image).unsqueeze(0)  # 增加 batch 維度
    return image.to(device)


# 4. 後處理函數
def postprocess_output(output, threshold=0.5):
    output = torch.sigmoid(output)  # 將輸出映射到 [0, 1]
    output = output.squeeze().cpu().numpy()  # 去除 batch 維度並轉為 numpy
    output_binary = (output > threshold).astype(np.uint8)  # 二值化處理
    return output, output_binary

# 5. 推論邏輯
def inference(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)  # 推論
    return output

# 6. 給予結果顏色處理，將白色區域視為免疫細胞
def create_colored_mask(binary_mask, chunk_size=1024):
    h, w = binary_mask.shape
    colored_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    # 逐塊處理
    for i in range(0, h, chunk_size):
        for j in range(0, w, chunk_size):
            # 計算當前塊的範圍
            h_end = min(i + chunk_size, h)
            w_end = min(j + chunk_size, w)

            # 提取當前區塊的二值 mask
            sub_mask = binary_mask[i:h_end, j:w_end]

            # 更新彩色 mask 對應區域
            colored_mask[i:h_end, j:w_end][sub_mask == 1] = [0, 0, 0]      # 背景區域（黑色）
            colored_mask[i:h_end, j:w_end][sub_mask == 0] = [255, 255, 255]  # 免疫細胞（白色）

    return colored_mask

def generate_binary_map(output_map, output_path, wsi_name):
    immune_binary_map = (output_map > 0).astype(np.uint8) * 255  # 轉換為 0 和 255
    immune_binary_map_path = os.path.join(output_path, f"{wsi_name}_immune_binary.png")
    cv2.imwrite(immune_binary_map_path, immune_binary_map)
    print(f"Immune binary image saved success!, \n圖片大小: {immune_binary_map.shape}")

def apply_mask_to_overlay(original_img, binary_mask):

    overlay_map = original_img.copy()
    overlay_map[binary_mask] = [255, 0, 0, 255]  # 白色區域標為紅色
    overlay_map_BGR = cv2.cvtColor(overlay_map, cv2.COLOR_RGBA2BGR)
    return overlay_map_BGR

# def apply_mask_to_overlay(original_img, binary_mask, alpha=0.5):
#     binary_mask = binary_mask.astype(bool)
#     colored_overlay = np.zeros_like(original_img)
#     colored_overlay[binary_mask] = [0, 0, 255]
#     overlay = cv2.addWeighted(original_img, 1 - alpha, colored_overlay, alpha, 0)
#     return overlay

    
def wsi_inference(wsi_path, model, params, output_path):
    """
    對整個 WSI 進行推論並生成 mask，加入組織檢測
    """
    os.makedirs(output_path, exist_ok=True)
    dataset = SlideDataset(wsi_path, params, preprocess=True, output_path=output_path)
    dataloader = data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)
    
    wsi_width, wsi_height = dataset.__size__()
    wsi_mask = np.zeros((wsi_height, wsi_width, 3), dtype=np.uint8)
    wsi_overlay = np.zeros((wsi_height, wsi_width, 3), dtype=np.uint8) * 255
    immune_map = np.zeros((wsi_height, wsi_width), dtype=np.uint8)  # 只存 0 和 255
    
    # 處理每個batch
    for batch in tqdm(dataloader, desc="處理 WSI Batches"):
        images, x_coords, y_coords = batch
        valid_indices = []
        valid_images = []
        valid_originals = []
        
        # 檢查每個patch是否包含組織
        for i, img in enumerate(images):
            img_np = img.numpy().astype(np.uint8)
            if dataset.has_tissue(img_np):
                valid_indices.append(i)
                valid_images.append(preprocess_image(img_np))
                valid_originals.append(img_np)
                
        if not valid_images:
            continue
            
        # 將有效的patches堆疊成batch
        valid_batch = torch.cat(valid_images, dim=0)
        
        # 進行推論
        with torch.no_grad():
            outputs = model(valid_batch)
            outputs = torch.sigmoid(outputs)
            binary_masks = (outputs > 0.5).cpu().numpy()
        
        # 處理每個有效的patch結果
        for idx, valid_idx in enumerate(valid_indices):
            binary_mask = binary_masks[idx]
            original_patch = valid_originals[idx]
            colored_mask = create_colored_mask(binary_mask[0], chunk_size=512)
            
            x, y = x_coords[valid_idx], y_coords[valid_idx]
            
            # 保存包含免疫細胞的patch
#             if np.any(colored_mask == 255):
#                 patch_filename = os.path.join(output_path, f'patch_{x}_{y}.png')
#                 cv2.imwrite(patch_filename, colored_mask)
            
            # 更新WSI mask
            wsi_mask[y:y+params['patch_size'], x:x+params['patch_size']] = colored_mask
            
            # 生成帶紅色標註的 overlay
            overlay_patch = apply_mask_to_overlay(original_patch, binary_mask[1])
            wsi_overlay[y:y + params['patch_size'], x:x + params['patch_size']] = overlay_patch
            
            # 二元免疫細胞 map
            immune_map[y:y+params['patch_size'], x:x+params['patch_size']] = binary_mask[1] * 255


    generate_binary_map(immune_map, output_path, os.path.basename(wsi_path).split('.')[0])

    # 印出原圖 + 標註區域
    overlay_output_path = os.path.join(output_path, f"{os.path.basename(wsi_path).split('.')[0]}_immune_only.png")
    cv2.imwrite(overlay_output_path, wsi_overlay)
    
    return 0

def main():
    params = {
        'patch_size': 512,
        'patch_overlap_ratio': 0,
        'image_rate': '20x',
        'batch_size': 10  # 添加batch_size參數
    }

#     weight_path = "/work/u2204495/Immune_Cell/train/weights/best_model_90.77.pt" # 亦豪的 weight
#     weight_path = "/work/u2204495/Immune_Cell/train/weights/best_model_78.54.pt" # 我自己 train 最好的
#     wsi_input_path = "/work/u2204495/liver_portal/wsi/test/22-00402-HE.mrxs"
#     output_path = "/work/u2204495/liver_portal/wsi/test/inference_result"

    # 從命令列參數讀取
    weight_path = args.weight_path
    wsi_input_path = args.wsi_input_path
    output_path = args.output_path
    
    model = load_model(weight_path, num_classes=2)
    mask_path = wsi_inference(wsi_input_path, model, params, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_input_path", type=str, required=True, help="Path to the WSI file")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save output results")
    args = parser.parse_args()
    main()
# endregion


