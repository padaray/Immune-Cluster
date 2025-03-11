import argparse
import cv2
import os
import pyvips
import pickle
import torch
import math
import numpy as np

from itertools import repeat
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from utils.tools import WSI_Reader
from utils.remove_dirt import remove_dirt

from timer import timer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_path", type=str, help="wsi saving path")
    parser.add_argument("--pkl_path", type=str, default=None, 
                        help="pikle file saving path")
    parser.add_argument("--patch_saving_dir", type=str, default=None, 
                        help="patch images saving path")
    parser.add_argument("--patch_wsi_mag", type=int, default=40, 
                        help="reading magnification in WSI to cut patch")
    parser.add_argument("--patch_size", "--ps", type=int, default=512, 
                        help="patch size of training data")
    parser.add_argument("--stride_size", "--ss", type=int, default=512, 
                        help="stirde step for slicing training data")
    parser.add_argument("--segment_output_dir", "--op", type=str, default=None, 
                        help="directory of saving output image of segment")
    parser.add_argument("--num_workers", default=os.cpu_count(),
                        help="the number of worker ")
    parser.add_argument("--clear_wsi_mag", default=2.5, 
                        help="read magnification to feed into the model for\
                        removing the dirt and marker.")
    return parser.parse_args()


@timer
def segment_tissue(wsi_path, clear_wsi_mag):
    """
    segment the tissue from wsi, and return clean tissue images 
    and masks with numpy array type
    """

    # 1. build WSI_reader object
    source_wsi_reader = WSI_Reader(wsi_path)

    # 2. read wsi under the magnification for feeding into removal model
    ori_source_mag_2d5 = source_wsi_reader.read_wsi(clear_wsi_mag)
    
    # 2. remove marker and dirt
    source_mask_mag_2d5 = remove_dirt(ori_source_mag_2d5)
    torch.cuda.empty_cache()

    # # 3. Gnerate the tissue mask
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  
    # source_mask_mag_2d5 = cv2.erode(source_mask_mag_2d5, kernel, iterations=3)
    # source_mask_mag_2d5 = cv2.dilate(source_mask_mag_2d5, kernel, iterations=2)

    # 4. Generate the clean tissue image with tissue mask
    clean_source_mag_2d5 = ori_source_mag_2d5.copy()
    clean_source_mag_2d5[source_mask_mag_2d5 < 125] = [255, 255, 255]
    
    return ori_source_mag_2d5, clean_source_mag_2d5, source_mask_mag_2d5


def get_patch_wsi_level(wsi_path:str, patch_wsi_mag):
    """
    return the reading level of the corresponding patch_wsi_mag in the WSI setting
    """
    wsi_reader = WSI_Reader(wsi_path)
    mag_dict = wsi_reader.get_mag_dict()
    patch_wsi_level = list(mag_dict.keys())[list(mag_dict.values()).index(patch_wsi_mag)]
    
    return int(patch_wsi_level)


@timer
def remove_non_liver(tissue_img, tissue_mask):
    """
    get numpy array type tissue image and mask, remove the non-liver region
    with oulier in mean color and return array type non-liver tissue images
    and masks
    """
    contours, hierarchy = cv2.findContours(tissue_mask, cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)  # find contour of tissue
                                           
    output = np.zeros_like(tissue_img)  # result image remove non-liver 
    output_mask = np.zeros_like(tissue_mask)  # reslut mask remove non-liver 
    tissue_mean_color = []  # store 3-chennal color for each tissue
    mask_save = []
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), 
                      reverse=True)  # sort tissue contour by region area

    for cnt in contours:
        single_tissue_mask = np.zeros_like(tissue_mask)  # mask for single tissue
        if cv2.contourArea(cnt) > CONTOUR_AREA_THRESHOLD:
            cv2.drawContours(single_tissue_mask, [cnt], -1, 255, thickness=-1)
            single_tissue_region = cv2.bitwise_and(tissue_img, tissue_img, 
                                                   mask=single_tissue_mask)
            mean_color = []  # store mean color for each channel in single tiuuse refgion
            for c in range(3):
                mean_tissue_region = single_tissue_region[:, :, c]
                mean_tissue_region = mean_tissue_region[mean_tissue_region != 0]
                mean_tissue_region = mean_tissue_region[mean_tissue_region < 200]
                mean_color.append(np.mean(mean_tissue_region)) 
            tissue_mean_color.append(mean_color)
            mask_save.append(single_tissue_mask)
    
    # Outlier detection
    tissue_mean_color = np.array(tissue_mean_color)
    Q2 = np.percentile(tissue_mean_color[:, 0], 50, interpolation='midpoint') 
    outlier = np.where((tissue_mean_color[:, 0] > Q2 * 1.05) | (tissue_mean_color[:, 0] < Q2 / 1.05))[0]  # Upper bound

    for _id, mask in enumerate(mask_save):
        if _id not in outlier:
            output += cv2.bitwise_and(tissue_img, tissue_img, mask=mask)
            output_mask += mask
    
    return output, output_mask


def save_tissue_images(saving_dir, case_name, tissue_images):
    """
    save images during tissue segmentation in designated directory
    """
    os.makedirs(saving_dir, exist_ok=True)
    cv2.imwrite(f"{saving_dir}/{case_name}.png", tissue_images['original_img'])
    cv2.imwrite(f"{saving_dir}/{case_name}_rm.png", tissue_images['tissue_img'])
    cv2.imwrite(f"{saving_dir}/{case_name}_mask.png", tissue_images['tissue_mask'])
    cv2.imwrite(f"{saving_dir}/{case_name}_output.png", 
                  tissue_images['non_liver_tissue_img'])
    cv2.imwrite(f"{saving_dir}/{case_name}_output_mask.png", 
                tissue_images['non_liver_tissue_mask'])
    print("Tissue Results Saved!")


def scale_coordinate(x, y, scale_factor):
    """
    covert the coordinates to the designated scale
    """
    x = int(x / pow(2, scale_factor))
    y = int(y / pow(2, scale_factor))
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y

    return x, y


def detect_background(img):
    """
    return true if image is foreground else return false
    """
    np.seterr(divide='ignore', invalid='ignore')
    sat = np.nan_to_num(
        1 - np.amin(img, axis=2) / np.amax(img, axis=2))
    pix_sat_count = (sat < 0.1).sum()
    all_pix_count = (sat > -1).sum()

    if pix_sat_count > (all_pix_count * 0.75):
        return False

    return True


def save_pkl_file(pkl_list, pkl_path):
    """
    save coordinates list as pikle file 
    """
    with open(pkl_path, 'ab') as pkl:
        pickle.dump(pkl_list, pkl)


def save_img(start_loc, img, patch_saving_dir, case_name):    
    x, y = start_loc
    cv2.imwrite(f'{patch_saving_dir}/{case_name}/{case_name}_{x}_{y}.png', img[:,:,::-1])

def run_cut_patch(patch_info_chunk, params):
    """
    get patch coordinate in wsi level 0, and return coordinate only if patch is
    in liver region and not background
    """

    slide = pyvips.Image.new_from_file(params['wsi_path'], level=params['level'])
    patch_size = params['patch_size']
    scaled_tissue_mask = params['scaled_tissue_mask']
    scale_power = math.log(patch_size, 2)
    corrdinates_of_interest = []
    images_of_interest = []

    if params['patch_saving_dir'] != None:
        os.makedirs(f"{params['patch_saving_dir']}/{params['case_name']}", exist_ok=True)

    for patch_info in patch_info_chunk:
        sx = int(patch_info['sx'])
        sy = int(patch_info['sy'])
        scaled_x, scaled_y = scale_coordinate(sx, sy, scale_power)
        slide_data = pyvips.Region.new(slide).fetch(sx, sy, patch_size, patch_size)
        img = np.ndarray(buffer=slide_data,
                        dtype=np.uint8,
                        shape=[patch_size, patch_size, params['bands']])

        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        
        try:
            # patch is not liver and not background
            if scaled_tissue_mask[scaled_y, scaled_x] > 0 and detect_background(img.copy()):  
                corrdinates_of_interest.append([sx, sy])
                images_of_interest.append(img)
        except Exception as e:
            print(e)
        
        if len(corrdinates_of_interest) > params['chunk_size']:
            
            # saving pickle image
            if params['pkl_path'] != None:
                with ThreadPoolExecutor() as e:
                    for r in e.map(save_pkl_file, corrdinates_of_interest, repeat(params['pkl_path'])):
                        if r is not None:
                            print(r)

            # saving patch image
            if params['patch_saving_dir'] != None:
                with ThreadPoolExecutor() as e:
                    for r in e.map(save_img, corrdinates_of_interest,
                                   images_of_interest,
                                   repeat(params['patch_saving_dir']),
                                   repeat(params['case_name'])):
                        if r is not None:
                            print(r)
            corrdinates_of_interest = []
            images_of_interest = []

    if len(corrdinates_of_interest):
        # saving pickle image
        if params['pkl_path'] != None:
            
            with ThreadPoolExecutor() as e:
                for r in e.map(save_pkl_file, corrdinates_of_interest, 
                               repeat(params['pkl_path'])):
                    if r is not None:
                        print(r)

        # saving patch image
        if params['patch_saving_dir'] != None:
            with ThreadPoolExecutor() as e:
                for r in e.map(save_img, corrdinates_of_interest,
                                images_of_interest,
                                repeat(params['patch_saving_dir']),
                                repeat(params['case_name'])):
                    if r is not None:
                        print(r)
        corrdinates_of_interest = []
        images_of_interest = []


@timer
def gen_pkl(args, non_liver_tissue_mask):
    """
    generate pikle list with coordinates of patches of interest
    """
    try:
        level = get_patch_wsi_level(args.wsi_path, args.patch_wsi_mag)
    except:
        level = 0
    try:
        slide = pyvips.Image.new_from_file(args.wsi_path, level=level)
    except:
        slide = pyvips.Image.new_from_file(args.wsi_path, page=level)
    patch_info_list = []
    mask_scale_factor = args.patch_size / (args.patch_wsi_mag / args.clear_wsi_mag)
    (mask_h, mask_w) = np.shape(non_liver_tissue_mask)
    scaled_tissue_mask = cv2.resize(non_liver_tissue_mask, 
                        (int(mask_w / mask_scale_factor), int(mask_h / mask_scale_factor)))     
    
    for sy in range(0, slide.height - args.patch_size, args.stride_size):
        for sx in range(0, slide.width - args.patch_size, args.stride_size):
            patch_info_list.append({'sx': sx,  # patch starting point x
                                    'sy': sy})
    
    chunk_size = len(patch_info_list) // args.num_workers
    patch_info_chunks = np.array_split(patch_info_list, chunk_size)
    params = dict(patch_size=args.patch_size, 
                  level=level,
                  wsi_path=args.wsi_path,
                  patch_saving_dir=args.patch_saving_dir,
                  case_name=os.path.basename(args.wsi_path).split('.')[0],
                  bands=slide.bands,
                  chunk_size=chunk_size,
                  pkl_path=args.pkl_path,
                  scaled_tissue_mask=scaled_tissue_mask)
    
    with ProcessPoolExecutor(args.num_workers) as e:
        for r in tqdm(e.map(run_cut_patch, patch_info_chunks, repeat(params)), 
                                        total=len(patch_info_chunks)):
            if r is not None:
                print(r)


if __name__ == "__main__":

    args = get_args()

    CONTOUR_AREA_THRESHOLD = 350000

    # Tissue Segmentation
    original_img, tissue_img, tissue_mask = segment_tissue(args.wsi_path, args.clear_wsi_mag)  # segment tissue
    # non_liver_tissue_img, non_liver_tissue_mask = remove_non_liver(tissue_img, 
    #                                                                tissue_mask)  # remove non-liver
    print("Tissue Segmentation Finished!")

    # Save Images and Masks
    if args.segment_output_dir != None:
        tissue_images = {
            "original_img": original_img,
            "tissue_img": tissue_img,
            "tissue_mask": tissue_mask,
            "non_liver_tissue_img": tissue_img,
            "non_liver_tissue_mask": tissue_mask
        }
        save_tissue_images(args.segment_output_dir, 
                           os.path.basename(args.wsi_path).split('.')[0], 
                           tissue_images)
        print(f"Tissue Images Saved!")

    # Remove the previous pickle file 
    if args.pkl_path != None:
        try:
            os.remove(args.pkl_path) 
        except:
            print("No previous pkl file.") 

        # Save Pikle File
        pkl_list = gen_pkl(args, tissue_mask)
        print(f"Pickle File '{args.pkl_path}' Saved!")
