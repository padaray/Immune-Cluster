# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import Dataset
import pyvips
import pickle
import cv2
import os
from PIL import ImageFile
from PIL import Image

import imgaug.augmenters as iaa


ALL_transform = iaa.Sequential([
                            iaa.Resize({"height": 256, "width": 256}),
                            ])

class immune_cell_dataset(Dataset):
    def __init__(self, data_params, transform):
        super().__init__()
        self.patch_size = data_params['patch_size']
        self.transform = transform
        self.wsi_path = data_params['wsi_path']
        
        # The post fix of the wsi file
        postfix = self.wsi_path.split('.')[-1]
    
        if(postfix == "mrxs"):
            self.slide = pyvips.Image.new_from_file(data_params['wsi_path'], level=data_params['level'])
        elif(postfix == "tif"):
            self.slide = pyvips.Image.new_from_file(data_params['wsi_path'], page=data_params['level'])
        elif(postfix == "svs"):
            self.slide = pyvips.Image.new_from_file(data_params['wsi_path'])
        else:
            print("Invalid WSI file format")
            return
            
        self.patch_list = self.load_data_pkl(data_params['pkl_path'])
        
    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        x = int(float(self.patch_list[idx][0]))  # data = [sx, sy]
        y = int(float(self.patch_list[idx][1]))
        img = self.read2patch(x, y)

        if img.shape[2] == 4:  # remove the 4th channel in the patch image
            img = img[:, :, 0:3]

        if self.transform is not None:
            img = self.transform(img)

        return img, x, y

    def read2patch(self, x, y):
        slide_region = pyvips.Region.new(self.slide)
        slide_fetch = slide_region.fetch(x, y, self.patch_size, self.patch_size)
        img = np.ndarray(buffer=slide_fetch,
                         dtype=np.uint8,
                         shape=[self.patch_size, self.patch_size, self.slide.bands])
        
        return img

    def load_data_pkl(self, pkl_path):
        data = []

        # with open(pkl_path, 'rb') as f:
        #     try:
        #         data.extend(pickle.load(f))
        #     except Exception as e:
        #         print(e)                    
        with open(pkl_path, 'rb') as f:
            while 1:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break

        print("Loading Pickle File Success!")

        return np.asarray(data)


class immune_cell_patch_dataset(Dataset):
    def __init__(self, data_params, transform):
        super().__init__()
        self.transform = transform
        self.patch_list = self.load_data_pkl(data_params['pkl_path'])

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        img_path = self.patch_list[idx]
        img = self.get_data(img_path)
        
        if img.shape[2] == 4:  # remove the 4th channel in the patch image
            img = img[:, :, 0:3]

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path

    def get_data(self, img_path):
        # Default cv2 read file
        try:
            img = cv2.imread(img_path)[...,::-1].copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # PIL instead when error
        except:
            print("EXCEPT")
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(img_path)
            img = np.asarray(img)
            
        # img = ALL_transform(image=img)
            
        return img
    
    def load_data_pkl(self, pkl_path):
        data = []
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
                
        print("Loading Pickle File Success!")

        return np.asarray(data)


class immune_cell_evaluate_dataset(Dataset):
    def __init__(self, img_folder, gt_folder, transform=None):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.transform = transform
        self.img_names = os.listdir(img_folder)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        gt_path = os.path.join(self.gt_folder, img_name)

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')

        if self.transform:
            img = self.transform(img)
            gt = self.transform(gt)

        return img, gt, img_name

