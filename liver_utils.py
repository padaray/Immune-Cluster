import os
vipshome = r'C:\vips-dev-8.11\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips as vi
import numpy as np
from torchvision import transforms

from torch.utils.data import Dataset
import torch
import math
import cv2


def get_level(img_path, image_rate):
    
    slide = vi.Image.new_from_file( img_path )
    res = slide.xres
    if res < 100.0:
        res = 8000.0
    
    print(f"image_rate is: {image_rate}")
    image_rate = float(image_rate.split('x')[0])
    
    level = round(math.log2(res / (image_rate*100)))
    print(f"Inference level is: {level}")
    
    return level

def get_region( region, width, height, patch_size, level  ):
    for i in range(4):
        region[i] = region[i] * math.pow(2, -level)
        region[i] = max( region[i], 0 )
    
    if region[0]>region[2]:
        region[0], region[2] = region[2], region[0]
    
    if region[1]>region[3]:
        region[1], region[3] = region[3], region[1]
    
    if region[2] - region[0] < patch_size:
        if region[0] + patch_size < width:
            region[2] = region[0] + patch_size
        else:
            region[0] = region[2] - patch_size
    
    if region[3] - region[1] < patch_size:
        if region[1] + patch_size < height:
            region[3] = region[1] + patch_size
        else:
            region[1] = region[3] - patch_size
    
    return region



class slide_ds(Dataset):
    def __init__(self, data_path, params, region=[-1, -1, -1, -1], preprocess=False, output_path=None):
        super().__init__()
        self.patch_size = params['patch_size']
        self.stride_size = int(self.patch_size * (1 - params['patch_overlap_ratio']))
        
        image_rate = params['image_rate']
        self.level = get_level(data_path, image_rate)
        self.preprocess = preprocess
        self.output_path = output_path

        try:
            self.slide = vi.Image.tiffload(data_path, page=self.level)
        except:
            try:
                self.slide = vi.Image.new_from_file(data_path, level=self.level)
            except:
                self.slide = vi.Image.new_from_file(data_path)
                self.slide = self.slide.resize(math.pow(2, -self.level))

        if region != [-1, -1, -1, -1]:
            region = get_region(region, self.slide.width, self.slide.height, self.patch_size, self.level)
            self.slide = self.slide.crop(int(region[0]), int(region[1]), int(region[2]-region[0]), int(region[3]-region[1]))
        else:
            region = [0, 0, self.slide.width, self.slide.height]
        
        self.region = region
        self.img_region = vi.Region.new(self.slide)
        self.patch_list = [(sx, sy)
                            for sy in range(0, self.slide.height-self.patch_size, self.stride_size)
                            for sx in range(0, self.slide.width-self.patch_size, self.stride_size)]

    ######## 確認圖片內有無組織 ########
    def get_tissue_mask(self, img, conn_thresh=1000):
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

    def has_tissue(self, img):
        tissue_mask = self.get_tissue_mask(img)
        tissue_ratio = np.sum(tissue_mask) / tissue_mask.size
        return tissue_ratio > 0.1  # 假設組織占比超過10%才視為有效patch
    
    def __len__(self):
        return len(self.patch_list)
    
    def __size__(self):
        return self.slide.width, self.slide.height

    def __getitem__(self, idx):
        (x, y) = self.patch_list[idx]
        patch_data = self.img_region.fetch(x, y, self.patch_size, self.patch_size)
        img = np.ndarray(buffer=patch_data, dtype=np.uint8, shape=[self.patch_size, self.patch_size, self.slide.bands])
        # 是否要預處理
        if self.preprocess:
            return img, x, y

        else: 
            if img.shape[2] == 4:
                img = img[:, :, :3]
                img = torch.tensor(img, dtype = torch.float32).permute(2,0,1) / 255
            return img, x, y     


    #----------------------------- 預處理 Function -----------------------------
    def preprocess_image(self, img, target_size=(256, 256)):
        # 確保是 numpy 陣列
        if not isinstance(img, np.ndarray):
            img = img.numpy() if torch.is_tensor(img) else np.array(img)

        # 確保是 RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[2] == 3 and img.dtype == np.uint8:
            # 如果是 BGR，轉換為 RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 調整大小
        img = cv2.resize(img, target_size)

        # 標準化轉換
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img)
        return img_tensor
    #-------------------------------------------------------------------------------
