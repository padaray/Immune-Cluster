# region
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from PIL import Image
import argparse
import openslide
from memory_profiler import memory_usage

Image.MAX_IMAGE_PIXELS = None

# 查看記憶體用量
def monitor_memory(func, func_name, *args, **kwargs):
    """監測函數執行時的記憶體使用變化（以 GB 為單位），不使用 psutil"""
    
    # 取得執行前的最小記憶體使用量
    mem_before = min(memory_usage(-1, interval=0.1, timeout=1))  # 以 MB 為單位
    
    # 監測函數執行期間的記憶體變化
    mem_usage, result = memory_usage((func, args, kwargs), retval=True, interval=0.1)
    
    # 取得執行後的記憶體使用量
    mem_after = max(memory_usage(-1, interval=0.1, timeout=1))  # 以 MB 為單位
    
    # 計算記憶體變化
    mem_peak = max(mem_usage) / 1024  # 轉換為 GB，最大記憶體用量
    mem_increase_gb = (max(mem_usage) - mem_before) / 1024  # 總共增加多少記憶體
    
    print(f"Memory usage increased by {mem_increase_gb:.4f} GB during {func_name}")
    print(f"Max memory usage during {func_name}: {mem_peak:.4f} GB")
    
    return result

# 移除和 portal 區域交集的免疫細胞
def filter_binary_maps(immune_map_path, portal_map_path):
    immune_map = np.array(Image.open(immune_map_path).convert("L"))
    portal_map = np.array(Image.open(portal_map_path).convert("L"))
    
    if immune_map.size == 0 or portal_map.size == 0:
        raise ValueError("影像讀取失敗或影像為空！")
    
    if immune_map.shape != portal_map.shape:
        raise ValueError("Immune 和 Portal 圖片大小不匹配")
    
    filtered_map = immune_map.copy()
    filtered_map[(immune_map == 255) & (portal_map == 255)] = 0
    
    return immune_map, filtered_map


def extract_white_regions(binary_map):
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.array([pt[0] for contour in contours for pt in contour])
    print(f"Total number of points in white regions: {len(points)}")
    return points


# 用 DBSCAN 演算法計算有多少個群聚
def apply_dbscan_clustering(points, eps=30, min_samples=40):
    if len(points) == 0:
        return np.array([]), np.array([])
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return points, labels

# ------------------------- 生成熱成像圖 -------------------------
def apply_heatmap_overlay(original_image, heatmap):
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 使用 JET colormap
    overlay = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)  # 疊加
    return overlay

def generate_heatmap(points, labels, image_shape, original_image, output_path, wsi_path):
    def inner_generate():
        x_bins = np.linspace(0, image_shape[1], 200)
        y_bins = np.linspace(0, image_shape[0], 80)
        density_map = np.zeros((len(y_bins) - 1, len(x_bins) - 1))

        unique_labels = set(labels)
        for label in unique_labels:
            if label != -1:
                cluster_points = points[labels == label]
                hist, _, _ = np.histogram2d(cluster_points[:, 1], cluster_points[:, 0], bins=[y_bins, x_bins])
                density_map += hist

        density_map = gaussian_filter(density_map, sigma=2)
        density_map = (density_map / np.max(density_map) * 255).astype(np.uint8)  # 正規化到 0-255

        # 調整熱力圖大小以匹配原始影像
        heatmap_resized = cv2.resize(density_map, (original_image.shape[1], original_image.shape[0]))
        overlayed_image = apply_heatmap_overlay(original_image, heatmap_resized)

        # 儲存結果
        output_filename = os.path.join(output_path, f"{os.path.basename(wsi_path).split('.')[0]}_heatmap.png")
        cv2.imwrite(output_filename, overlayed_image)
        print("generate_heatmap Success!")

    monitor_memory(inner_generate, "generate_heatmap")
# -----------------------------------------------------------
    
    
# ------------------ 生成 WSI 把標註免疫細胞塗紅 ------------------
def generate_filtered_overlay(original_image, filtered_map, output_path, wsi_path, suffix="filtered"):
    def inner_generate():
        overlay_map = original_image.copy()
        
        # 找細胞輪廓
        contours, _ = cv2.findContours(filtered_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(overlay_map, (cx, cy), 3, (255, 0, 0), -1)  # 只標註中心點

        overlay_output_path = os.path.join(output_path, f"{os.path.basename(wsi_path).split('.')[0]}_clust_immune_{suffix}.png")
        overlay_map = cv2.cvtColor(overlay_map, cv2.COLOR_BGR2RGB)
        cv2.imwrite(overlay_output_path, overlay_map)
        print(f"generate_{suffix}_overlay Success!")
        
    monitor_memory(inner_generate, "generate_filtered_overlay")
# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--immune_map", type=str, required=True, help="免疫細胞二元圖像路徑")
    parser.add_argument("--portal_map", type=str, required=True, help="Portal區域二元圖像路徑")
    parser.add_argument("--wsi_path", type=str, required=True, help="原始 mrxs 圖像路徑")
    parser.add_argument("--output_path", type=str, required=True, help="輸出的熱力圖圖像路徑")
    args = parser.parse_args()
    
    immune_map, filtered_map = filter_binary_maps(args.immune_map, args.portal_map)
    
    immune_cells = extract_white_regions(filtered_map)
    points, labels = apply_dbscan_clustering(immune_cells)
    
    slide = openslide.OpenSlide(args.wsi_path)
    original_image = np.array(slide.get_thumbnail((filtered_map.shape[1], filtered_map.shape[0])))
    generate_heatmap(points, labels, filtered_map.shape, original_image, args.output_path, args.wsi_path)
    
    mem_before = memory_usage()[0]
    generate_filtered_overlay(original_image, filtered_map, args.output_path, args.wsi_path, "filtered")
    generate_filtered_overlay(original_image, immune_map, args.output_path, args.wsi_path, "original")

# endregion


