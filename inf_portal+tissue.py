import os
import numpy as np
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset, DataLoader
import json
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
from liver_utils import slide_ds
import pyvips as vi
from scipy import ndimage as ndi
import cv2
from collections import OrderedDict
from scipy.sparse import coo_matrix
import argparse
import gc


# +
class Algorithm():
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        
        # 直接從命令列參數讀取
        self.params = {
            "wsi_path": args.wsi_path,
            "patch_size": args.patch_size,
            "patch_overlap_ratio": args.patch_overlap_ratio,
            "batch_size": args.batch_size,
            "output_type": args.output_type,
            "model_dir": args.model_dir,
            "output_dir": args.output_dir
        }
        

    def init_parameters(self):
        
        
        ### 載入模型相關資訊
        self.model_portal_info = torch.load(self.params['model_dir'])
        print(f"portal model path： {self.params['model_dir']}")
        self.params['image_rate'] = self.model_portal_info['params']['image_rate']
        self.params['net_portal_name'] = self.model_portal_info['params']['net_name']
#         self.params['image_rate'] = "20x"
#         self.params['net_portal_name'] = "MAnet"
        
        ### outputs
        output_dir = self.params['output_dir'] #'./output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.label_path = output_dir + '/Steatosis_' # + filename.json
        
        ### time
        time_now = time.localtime()
        self.start_time = time.strftime('%Y%m%d_%H%M%S', time_now)

        
        if self.params['patch_overlap_ratio'] > 0:
            self.min_box = int(self.params['patch_size'] * self.params['patch_overlap_ratio'] / 2)
            self.max_box = self.params['patch_size'] - self.min_box
            self.scalar = torch.arange(self.params['patch_size'], device=self.device).unsqueeze(0)
        
        
    
    def load_model(self, model_info, net_name):
        if net_name == 'Unet':
            from UNet import UNET as Network
            model = Network(out_channels = 1+1)
        
        elif net_name == 'Unet_ResNet50':
            from model.UNet_ResNet50 import UNET_ResNet50 as Network
            model = Network(out_channels = 1+1)
            
        elif net_name == 'Unet_ASPP':
            from model.UNet_ASPP import UNET_ASPP as Network
            model = Network(out_channels = 1+1)
            
        elif net_name == 'AutoEncoder':
            from model.AutoEncoder import AE_FCN as Network
            model = Network(out_channels = 1+1)
            
        elif net_name == 'AutoEncoder_ResNet50':
            from model.AutoEncoder_ResNet50 import AE_ResNet50 as Network
            model = Network(out_channels = 1+1)
        
        elif net_name == 'MAnet':
            model = smp.MAnet(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=2                      # model output channels (number of classes in your dataset)
            )
        
        elif net_name == 'FPN':
            model = smp.FPN(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2)

        elif net_name == 'Linknet':
            model = smp.Linknet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2)
        
        
        new_ck_pt = OrderedDict()

        try:
            ck_pt = model_info['model_state_dict']
        except:
            ck_pt = model_info

        for k, v in ck_pt.items():
            if 'module.features.' in k:
                k = k.replace('module.features.', 'features.module.')
            elif 'module.' in k:
                k = k.split('module.')[-1]
            new_ck_pt[k]=v

        model.load_state_dict(new_ck_pt)
        model = torch.nn.DataParallel(model).to(self.device)
        model.eval()
        return model
 

    def execute(self):
        
        self.model_portal = self.load_model(self.model_portal_info, self.params['net_portal_name'])

        self.execute_a_slide(self.params['wsi_path'])
        
    def execute_a_slide(self, img_path):
        
        print(f"wsi path: {img_path}")

        self.patch_count = 0
        self.results_infos = {}
        
        portal_dataset = slide_ds(img_path, params = self.params)
        
        self.total_patch = portal_dataset.__len__()
        
        slide_w, slide_h = portal_dataset.__size__()

        output_map = np.zeros([slide_h, slide_w], dtype=np.uint8)  # 初始化二維矩陣，矩陣內儲存 0~255
        tissue_only_map = np.zeros([slide_h, slide_w], dtype=np.uint8)  # 只有包含 tissue 的 map
        
        portal_loader = DataLoader(portal_dataset, batch_size=self.params['batch_size'], 
                                    shuffle=False, pin_memory=False, num_workers=0, drop_last=True)
        

        # ------------------------------ 推論 組織區域 和 Portal 區域 ------------------------------
        for imgs, xs, ys in tqdm(portal_loader):
            self.patch_count += imgs.shape[0]
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():

                    # Portal 預測
                    portal_result = self.model_portal(imgs.to(self.device))
                    portal_result = portal_result.cpu().numpy()
                    portal_result = (portal_result - np.max(portal_result, axis=1, keepdims=True)) >= 0
                    portal_result = portal_result[:,0,:,:]
                    portal_result = portal_result.astype(np.uint8)
                    portal_result[portal_result > 0] = 2  # Portal 類別設為 2

                    for b in range(portal_result.shape[0]):
                        x, y = xs[b], ys[b]

                        # 設置上、下、左、右邊界
                        left = 0 if x <= 0 else self.min_box
                        up = 0 if y <= 0 else self.min_box
                        right = output_map.shape[-1]-x if x+self.params['patch_size'] > output_map.shape[-1] else self.max_box
                        down = output_map.shape[-2]-y if y+self.params['patch_size'] > output_map.shape[-2] else self.max_box
                        
                        # 獲取組織 mask
                        tissue_patch = imgs[b].cpu().numpy().transpose(1, 2, 0)
                        tissue_patch = (tissue_patch * 255).astype(np.uint8)
                        tissue_mask = self.get_tissue_mask(tissue_patch, 20)
                        current_result = tissue_mask.astype(np.uint8) # 轉換為 uint8 類型，True = 1，False = 0
                        tissue_only_map[y+up:y+down, x+left:x+right] = current_result[up:down, left:right]
                        
                        # 組合 tissue 和 portal：portal 優先級高於 tissue
                        portal_and_tissue = np.logical_and(tissue_mask, portal_result[b] > 0)
                        current_result[portal_and_tissue] = 2

                        # 更新 output map
                        output_map[y+up:y+down, x+left:x+right] = current_result[up:down, left:right]
                        
                        
        ################### 計算連通區域並過濾小面積區域 ###################
        portal_mask = (output_map == 2) # 只處理 portal 連通區域
        labeled_map, num_features = ndi.label(portal_mask)
        area_threshold = 3000
        print("計算面積中......")

        region_areas = np.bincount(labeled_map.ravel())
        small_regions = np.where(region_areas < area_threshold)[0]
        
        # 將小的 portal 區域設為 tissue (類別 1)
        portal_to_remove = np.isin(labeled_map, small_regions)
        output_map[portal_to_remove & (output_map == 2)] = 1
        
        print("計算面積並刪除較小區域，完成！！！！")
        ######################################################################
        
        
        ############# Portal 和 Tissue 的 map #############
        portal_tissue_map = output_map.copy()
        
        ############# 根據 output_map 快速取得 tissue binary map（含 portal）#############
#         tissue_binary_map = ((portal_tissue_map == 1) | (portal_tissue_map == 2)).astype(np.uint8) * 255
#         tissue_binary_map = cv2.resize(tissue_binary_map, 
#                                        (tissue_binary_map.shape[1] * 2, tissue_binary_map.shape[0] * 2), 
#                                        interpolation=cv2.INTER_NEAREST)

#         tissue_binary_map_path = os.path.join(self.params['output_dir'], img_path.split('/')[-1].split('.')[0] + '_tissue_binary.png')
#         cv2.imwrite(tissue_binary_map_path, tissue_binary_map)
#         print(f"Tissue binary image saved (from output_map)!\n圖片大小: {tissue_binary_map.shape}")
        
        ############# 只有 portal 區域的 map #############
        portal_only_map = portal_tissue_map.copy()
        portal_only_map[portal_only_map == 1] = 0
        
        ############# 只有 portal 區域的二元 map #############
        portal_binary_map = (portal_only_map == 2).astype(np.uint8) * 255
        portal_binary_map = cv2.resize(portal_binary_map, (portal_binary_map.shape[1] * 2, portal_binary_map.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        portal_binary_map_path = os.path.join(self.params['output_dir'], img_path.split('/')[-1].split('.')[0] + '_portal_binary.png')
        cv2.imwrite(portal_binary_map_path, portal_binary_map)
        print(f"Portal binary image save success!\n圖片大小: {portal_binary_map.shape}")
        
        # ------------------------------ 推論 組織區域 和 Portal 區域 ------------------------------

        img = np.zeros((slide_h, slide_w, 3), dtype=np.uint8)
        wsi_image = vi.Image.new_from_file(img_path, access="sequential")
        chunk_size = 32768
        # 逐區塊讀取 WSI，並存入 NumPy 陣列
        for y in range(0, slide_h, chunk_size):
            for x in range(0, slide_w, chunk_size):
                # 讀取 WSI 影像的一個區塊
                img_chunk = self.load_wsi_tile(wsi_image, x, y, min(chunk_size, slide_w - x), min(chunk_size, slide_h - y))

                # 存入影像陣列
                img[y:y+img_chunk.shape[0], x:x+img_chunk.shape[1], :] = img_chunk
    
        
        ### write result to img
        result_p = self.params['output_dir'] + '/' + img_path.split('/')[-1].split('.')[0]

#         # 著色並儲存每張 map
#         portal_tissue_colored = self.color_map(img, portal_tissue_map)
#         cv2.imwrite(result_p + '_portal_tissue.png', portal_tissue_colored)
#         print("print portal & tissue image success")
        
#         del portal_tissue_colored
#         gc.collect()  # 強制執行垃圾回收
        
#         portal_only_colored = self.color_map(img, portal_only_map)
#         cv2.imwrite(result_p + '_portal_only.png', portal_only_colored)    
#         print("print portal only image success")

        
        return 0
    


# +
    #-------------------------- Map 著色 Function --------------------------
    def color_map(self, img, input_map, chunk_size=16384, color_sat=0.2):
        # 定義類別對應的顏色
        class_colors = {
            1: np.array([0, 0, 127], dtype=np.float32),  # 類別1的顏色
            2: np.array([0, 127, 0], dtype=np.float32)   # 類別2的顏色
        }

#         colored_img = np.zeros_like(img, dtype=np.uint8)
        colored_img = img.copy()

        # 逐塊處理影像和標記圖
        for y in range(0, img.shape[0], chunk_size):
            for x in range(0, img.shape[1], chunk_size):
                # 計算當前區塊的大小
                current_chunk_size_x = min(chunk_size, img.shape[1] - x)
                current_chunk_size_y = min(chunk_size, img.shape[0] - y)

                # 提取當前區塊的影像和標記圖
                img_chunk = img[y:y + current_chunk_size_y, x:x + current_chunk_size_x].astype(np.float32, copy=False)
                map_chunk = input_map[y:y + current_chunk_size_y, x:x + current_chunk_size_x]

                # 遍歷每個類別，應用對應的顏色
                for class_id, color in class_colors.items():
                    mask = (map_chunk == class_id)
                    img_chunk[mask] = img_chunk[mask] * (1 - color_sat) + color * color_sat

                # 將處理後的區塊放回著色後的影像中
                colored_img[y:y + current_chunk_size_y, x:x + current_chunk_size_x] = img_chunk.astype(np.uint8)

        return colored_img
    #----------------------------------------------------------------------------
    
    
    
    #--------------------------- 讀取圖片 Function ---------------------------
    def load_wsi_tile(self, wsi_image, x, y, width, height):

        # 讀取影像的一個區塊
        tile = wsi_image.crop(x, y, width, height)

        # 轉換為 NumPy 格式
        img_chunk = np.ndarray(
            buffer=tile.write_to_memory(),
            dtype=np.uint8,
            shape=(tile.height, tile.width, tile.bands)
        )

        # 確保影像只有 3 個通道 (RGB)
        if img_chunk.shape[2] >= 4:
            img_chunk = img_chunk[:, :, :3]

        return img_chunk
    #----------------------------------------------------------------------------
    
    
    
    #--------------------------- 抓組織的部分 Function ---------------------------
    def get_tissue_mask(self, img, conn_thresh):
        # 轉換成灰階
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 應用閾值
        mask[mask > 240] = 0
        mask[mask > 0] = 255
        
        # 反轉遮罩，使黑色區域變為前景
        inverted_mask = cv2.bitwise_not(mask)
        
        # 找出連通部分
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
        
        # 遍歷每個成分並根據面積過濾
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < conn_thresh:  # 移除小的黑色區域
                inverted_mask[labels == i] = 0  # 將這些區域設為白色（背景）
        
        # 將遮罩再反轉回來
        processed_mask = cv2.bitwise_not(inverted_mask).astype(bool)
        
        return processed_mask
    #-----------------------------------------------------------------------------



    #------------------- 取得 mpp(micrometer per pixel) Function -------------------
    def get_MPP(self, img_path, target_level):
        image = vi.Image.new_from_file(img_path)

        # 取得 MPP 資訊
        mpp_x = float(image.get("openslide.mpp-x") or 0)
        mpp_y = float(image.get("openslide.mpp-y") or 0)

        if not mpp_x or not mpp_y:
            print("無法取得 MPP 資訊。")
            return None, None

        print(f"在 level 0 時：")
        print(f"MPP (micrometer per pixel) x-axis：{mpp_x:.6f} µm | y-axis：{mpp_y:.6f} µm")

        # 顯示 level 1 到 level 4 的 MPP 資訊
        for level in range(1, 5):
            downsample = float(image.get(f"openslide.level[{level}].downsample") or 2 ** level)
            mpp_x_level = mpp_x * downsample
            mpp_y_level = mpp_y * downsample
            print(f"在 level {level} 時：")
            print(f"MPP (micrometer per pixel) x-axis：{mpp_x_level:.6f} µm | y-axis：{mpp_y_level:.6f} µm")

        # 針對指定 target_level 顯示 MPP 資訊
        print(f"\n--- 針對指定 level {target_level} 的 MPP 資訊 ---")
        target_downsample = float(image.get(f"openslide.level[{target_level}].downsample") or 2 ** target_level)
        mpp_x_target = mpp_x * target_downsample
        mpp_y_target = mpp_y * target_downsample
        print(f"MPP (micrometer per pixel) at level {target_level}：x-axis = {mpp_x_target:.6f} µm, y-axis = {mpp_y_target:.6f} µm")

        return mpp_x_target, mpp_y_target
    #-------------------------------------------------------------------------
# -

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_path", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--patch_overlap_ratio", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_type", type=str, default="polygon")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    Algorithm_runner = Algorithm()
    Algorithm_runner.init_parameters()
    Algorithm_runner.execute()




