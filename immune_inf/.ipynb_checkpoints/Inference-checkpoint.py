import argparse
import cv2
import json
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import segmentation_models_pytorch as smp
import pyvips
import openslide

from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
from dataset import immune_cell_dataset
from postprocess_functions import *
from preprocess import get_patch_wsi_level
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_path", type=str, help="full path of wsi file")
    parser.add_argument("--pkl_path", type=str, help="full path of pikle file")
    parser.add_argument("--inference_dir", type=str, default=None,
                        help="directory to save images during training")
    parser.add_argument("--immune_cell_weights", type=str, 
                        help="weight file for immune cell model to load")
    parser.add_argument("--cell_model", type=str, choices=['RU++', 'smp'],
                        help="immune cell segmentation model")
    parser.add_argument("--patch_size", "--pz", type=int, default=512,
                        help="inference patch size")
    parser.add_argument("--num_class", type=int, default=1,
                        help="number of class to feed in model")
    parser.add_argument("--batch_size", "--bz", type=int,
                        help="inference batch size")
    parser.add_argument("--patch_wsi_mag", type=int, default=40, 
                        help="reading magnification in WSI to cut patch")
    parser.add_argument("--save_mask", type=bool, default=False,
                        help="save patchwise mask")
    parser.add_argument("--save_all_cells", type=bool, default=False,
                        help="save patches with all cells contours")
    parser.add_argument("--filter_size", type=int, default=5,
                        help="Contour area lower than setting pixels is filtered ")
    
    
    return parser.parse_args()


def postprocess(img, patch_x, patch_y, masks_pred, level):
    args = get_args()
    wsi_case = os.path.basename(args.wsi_path).split('.')[0]
    img = img.detach().cpu().numpy()
    patch_x = patch_x.detach().cpu().numpy()
    patch_y = patch_y.detach().cpu().numpy()

    # Process Watershed Algorithm
    # gradient_mask = np.squeeze(masks_pred - border_pred)
    gradient_mask = np.squeeze(masks_pred)

    all_cells_mask = watershed(gradient_mask, 
                               np.squeeze(masks_pred > .5).astype(np.uint8),
                               (gradient_mask > .5).astype(np.uint8))
    all_cells_mask = (all_cells_mask > .5).astype(np.uint8)
    
    ############# 回傳用的座標和 mask #############
    useful_mask = (all_cells_mask > .5).astype(np.uint8) * 255
#     patch_x_1 = int(patch_x * pow(2, level))
#     patch_y_1 = int(patch_y * pow(2, level))
    ##############################################

    
    # Mask to contour
    all_cells_contours = filter_mask_2_contour(all_cells_mask, offsets=(0, 0))
    result_contours = all_cells_contours

    # Transfer Image/Mask into Numpy Array Type
    max_value = img.max()
    image = img * 255 / max_value
    image = np.uint8(image)
    image = image.transpose(1,2,0)[:,:,::-1].copy()
    
    

    mask = np.zeros((all_cells_mask.shape[0], all_cells_mask.shape[1], 3))
    mask[all_cells_mask == 1] = [255, 255, 255]
    
    # Calculate the mask area
    mask_area = np.sum(all_cells_mask == 1)

    # Save Predction Mask per Patches
    if args.save_mask:
        os.makedirs(f"{args.inference_dir}/MASK/" +\
                    f"{wsi_case}/", exist_ok=True)
        # print(args.save_mask)
        cv2.imwrite(f"{args.inference_dir}/MASK/" +\
                    f"{wsi_case}/{wsi_case}_{int(patch_x)}_{int(patch_y)}.png", 
                    mask)

    # Save Predction All Cells Contours per Patches 
    if args.save_all_cells:
        saving_img =  image.copy()
        cv2.drawContours(saving_img, all_cells_contours, -1, (127, 255, 212), 
                        thickness=2)
        os.makedirs(f"{args.inference_dir}/All_Cells/" +\
                    f"{wsi_case}/", exist_ok=True)
        cv2.imwrite(f"{args.inference_dir}/All_Cells/" +\
                    f"{wsi_case}/{wsi_case}_{int(patch_x)}_{int(patch_y)}.png", 
                    saving_img)


    # Create Point Annotation in WSI level 0
    points_annos = create_point_annotation((args.patch_size, args.patch_size), 
                                           result_contours, 
                                           int(patch_x), int(patch_y), args.filter_size)
    patch_x *= pow(2, level)
    patch_y *= pow(2, level)
    # scale coordinate to be level 0
    for coord in points_annos:
        coord['contour'] = [[c[0] * pow(2, level), c[1] * pow(2, level)] \
                            for c in coord['contour']]
        coord['x'] = coord['x'] * pow(2, level)
        coord['y'] = coord['y'] * pow(2, level)

    cell_annos_in_patch = dict(coordinates=points_annos, case=wsi_case, 
                               patch_x=int(patch_x), patch_y=int(patch_y))
    
    return cell_annos_in_patch, mask_area, useful_mask, int(patch_x), int(patch_y)


def save2json(annotation_list, inference_dir, version, wsi_case):
    BoundingBox_JSON = dict()
    BoundingBox_JSON['annotation'] = annotation_list
    BoundingBox_JSON['information'] = dict(name="immune cell", verison=version)
    os.makedirs(f"{inference_dir}/{version}/JSON/{wsi_case}", exist_ok=True)
    
    result_json_path = f"{inference_dir}/{version}/JSON/" + \
                       f"{wsi_case}/raw_annotation.json"
    with open(result_json_path, 'w') as f:
        json.dump(BoundingBox_JSON, f, indent=4)
         
    return result_json_path

def extract_white_regions(binary_map):
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.array([pt[0] for contour in contours for pt in contour])
    print(f"Total number of points in white regions: {len(points)}")
    return points


def predict_mask_process(mask):
    mask = torch.sigmoid(mask)
    binary_mask = torch.where(mask > 0.5, 1.0, 0.0)
    binary_masks_pred = torch.unsqueeze(binary_mask[:,0], dim=1)

    return  binary_masks_pred.detach().cpu().numpy()

@torch.no_grad()
def forward_step(model, imgs, device):
    imgs = imgs.to(device)

    if args.cell_model == "RU++":
        masks_pred, _ = model(imgs)
    else:
        masks_pred = model(imgs) 
    
    masks_pred = masks_pred[:, 1, :, :].unsqueeze(1)
    masks_pred = torch.sigmoid(masks_pred).detach().cpu().numpy()
    return masks_pred


# +
if __name__ == "__main__":
    # 1. Initital Setting=======================================================
    torch.manual_seed(0)
    args = get_args()    
    
    try:
        level = get_patch_wsi_level(args.wsi_path, args.patch_wsi_mag)
    except:
        level = 0
        
    ############################################################################
    slide_os = openslide.OpenSlide(args.wsi_path)

    if "aperio.AppMag" in slide_os.properties:
        base_mag = float(slide_os.properties["aperio.AppMag"])
    else:
        base_mag = 40.0

    downsample = slide_os.level_downsamples[level]
    actual_patch_mag = base_mag / downsample

    print(f"🧭 實際使用的 patch level = {level}，對應實際倍率 ≈ {actual_patch_mag:.2f}x")
    ############################################################################
        
    data_params = {
        'wsi_path':     args.wsi_path,
        'pkl_path':     args.pkl_path,
        'patch_size':   args.patch_size,
        'batch_size':   args.batch_size,
        'level':        level,
    }     

    num_class = args.num_class
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    if args.cell_model == "smp":
        model = smp.MAnet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=args.num_class,                      # model output channels (number of classes in your dataset)
        )
        
    else:
        print("Model Loading Error!")
        exit()

    model.to(device)
    
    # 讀取 WSI 原始大小
    slide = pyvips.Image.new_from_file(args.wsi_path)
    wsi_width, wsi_height = slide.width, slide.height

    # 初始化完整 WSI mask
    global_mask = np.zeros((wsi_height, wsi_width), dtype=np.uint8)

    if args.inference_dir != None:
        os.makedirs(args.inference_dir, exist_ok=True)

    # ==========================================================================
        
    # 2. Load Model Weights=====================================================
    checkpoint = torch.load(args.immune_cell_weights , map_location="cuda:0")

    try:
        if args.cell_model == "RU++":
            model.final = nn.Conv2d(64, args.num_class, kernel_size=1, stride=1, padding=0)
            
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint) 
            
        model.to(device)
        
    except:
        model.load_state_dict({k.replace('module.', ''): v 
                            for k, v in checkpoint['model_state_dict'].items()})

    model.to(device)
    # ==========================================================================

    # 3. Build Dataloader and Load into Model===================================
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    test_dataset = immune_cell_dataset(data_params, test_transform)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=args.batch_size, pin_memory=False, 
                                num_workers=1, drop_last=False)
#     print("Data Load Success!")
    # ==========================================================================

    # 4. Inference & Post-process===============================================
    cell_annos_results = []
    
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        imgs, patch_x, patch_y = batch
        with torch.cuda.amp.autocast(enabled=False):

            masks_preds = forward_step(model, imgs, device)

            with ThreadPoolExecutor(max_workers=1000) as e:
                results = e.map(postprocess, imgs, patch_x, patch_y, masks_preds, repeat(level))
                for annos, updated_mask, useful_mask, patch_x, patch_y in results:
                    cell_annos_results.append(annos)
                    global_mask[patch_y:patch_y+args.patch_size, patch_x:patch_x+args.patch_size] = useful_mask
                    
    # 儲存完整 WSI mask
#     H2, W2 = global_mask.shape[0] // 4, global_mask.shape[1] // 4
#     global_mask = cv2.resize(global_mask, (W2, H2), interpolation=cv2.INTER_NEAREST)
#     global_mask = (global_mask > 0).astype(np.uint8) * 255
    
#     save_filename = f"{os.path.basename(args.wsi_path).split('.')[0]}_immune_binary.png"
#     save_binary_path = os.path.join(args.inference_dir, save_filename)
#     cv2.imwrite(save_binary_path, global_mask)

#     print(f"Immune binary image save success!\n圖片大小: {global_mask.shape}")

    level_used = 2
    w, h = slide_os.level_dimensions[level_used]
    global_mask = cv2.resize(global_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    global_mask = (global_mask > 0).astype(np.uint8) * 255

    save_filename = f"{os.path.basename(args.wsi_path).split('.')[0]}_immune_binary.png"
    save_binary_path = os.path.join(args.inference_dir, save_filename)
    cv2.imwrite(save_binary_path, global_mask)
    
    print(f"Immune binary image save success!\n圖片大小: {global_mask.shape}")
    

    # ==========================================================================
    
    # 5. 疊加 global_mask_resized 到原圖縮圖 ====================
    print("Generating overlay on WSI at 10x resolution...")

    print(f"[INFO] Level 2 WSI 原始大小: width={w}, height={h}")
    
    wsi_level_img = slide_os.read_region((0, 0), level_used, (w, h)).convert("RGB")
    wsi_level_np = np.array(wsi_level_img)
    wsi_level_np = cv2.cvtColor(wsi_level_np, cv2.COLOR_RGB2BGR)
    
    # 確保尺寸一致，不需 resize
    assert global_mask.shape[:2] == wsi_level_np.shape[:2], "Mask and WSI dimensions do not match!"

    red_mask = np.zeros_like(wsi_level_np)
    red_mask[global_mask == 255] = [0, 0, 255]

    overlay_result = cv2.addWeighted(wsi_level_np, 0.6, red_mask, 0.4, 0)
    overlay_out_path = os.path.join(args.inference_dir, f"{os.path.basename(args.wsi_path).split('.')[0]}_immune_overlay_lvl{level_used}.png")
    cv2.imwrite(overlay_out_path, overlay_result)
    print(f"Overlay image save success!\n圖片大小: {overlay_result.shape}")
    # ===========================================================

