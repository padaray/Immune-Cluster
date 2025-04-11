# region
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from PIL import Image
import argparse
import openslide
from memory_profiler import memory_usage

Image.MAX_IMAGE_PIXELS = None

# ------------------ Memory Monitor ------------------
def monitor_memory(func, func_name, *args, **kwargs):
    mem_before = min(memory_usage(-1, interval=0.1, timeout=1))
    mem_usage, result = memory_usage((func, args, kwargs), retval=True, interval=0.1)
    mem_after = max(memory_usage(-1, interval=0.1, timeout=1))
    mem_peak = max(mem_usage) / 1024
    mem_increase_gb = (max(mem_usage) - mem_before) / 1024
    print(f"Memory usage increased by {mem_increase_gb:.4f} GB during {func_name}")
    print(f"Max memory usage during {func_name}: {mem_peak:.4f} GB")
    return result

# ------------------ Binary Map Filter ------------------
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

# ------------------ Extract Points ------------------
def extract_white_regions(binary_map):
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append([cx, cy])

    points = np.array(centers)
    print(f"Total number of immune cells (center points): {len(points)}")
    return points

# ------------------ DBSCAN ------------------
def apply_dbscan_clustering(points, eps=120, min_samples=10): #(50, 10)
    if len(points) == 0:
        return np.array([]), np.array([])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return points, labels

# ------------------ Heatmap Overlay ------------------
def apply_heatmap_overlay(original_image, heatmap):
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# ------------------ Generate Heatmap and Boxes ------------------
def generate_heatmap_and_boxes(points, labels, image_shape, original_image, output_path, wsi_path,
                                density_threshold, filtered_map, micrometer_per_pixel):
    def inner_generate():
        
        # 1. 計算密度圖
        desired_bin_count = 180
        height, width = image_shape
        aspect_ratio = height / width

        x_bins = np.linspace(0, width, desired_bin_count)
        y_bins = np.linspace(0, height, int(desired_bin_count * aspect_ratio))
        density_map = np.zeros((len(y_bins) - 1, len(x_bins) - 1))

        unique_labels = set(labels)
        for label in unique_labels:
            if label != -1:
                cluster_points = points[labels == label]
                hist, _, _ = np.histogram2d(cluster_points[:, 1], cluster_points[:, 0], bins=[y_bins, x_bins])
                density_map += hist

        density_map = gaussian_filter(density_map, sigma=2)
        density_map = (density_map / np.max(density_map) * 255).astype(np.uint8)

        # 2. 熱力圖輸出
        heatmap_resized = cv2.resize(density_map, (original_image.shape[1], original_image.shape[0]))
        overlayed_image = apply_heatmap_overlay(original_image, heatmap_resized)

        heatmap_filename = os.path.join(output_path, f"{os.path.basename(wsi_path).split('.')[0]}_heatmap.png")
        cv2.imwrite(heatmap_filename, overlayed_image)
        print("Heatmap saved to", heatmap_filename)

        # 3. 密度區塊框選 + 計算面積
        _, binary_mask = cv2.threshold(heatmap_resized, density_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay_image = original_image.copy()
        cluster_info = []

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 500:
                cv2.polylines(overlay_image, [contour], isClosed=True, color=(255, 0, 0), thickness=5)

                x, y, w, h = cv2.boundingRect(contour)
                immune_roi = filtered_map[y:y+h, x:x+w]
                immune_pixel_count = cv2.countNonZero(immune_roi)
                immune_area_um2 = immune_pixel_count * (micrometer_per_pixel ** 2)

                cluster_info.append({
                    "cluster_id": idx,
                    "bounding_box": [x, y, w, h],
                    "immune_pixels": immune_pixel_count,
                    "immune_area_um2": immune_area_um2
                })

        box_filename = os.path.join(output_path, f"{os.path.basename(wsi_path).split('.')[0]}_density_boxes.png")
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(box_filename, overlay_image)
        print("Box overlay saved to", box_filename)

        # 印出總結
        print("\n===== Cluster Area Summary =====")
        for info in cluster_info:
            print(f"Cluster {info['cluster_id']} | BBox: {info['bounding_box']} | Immune Pixels: {info['immune_pixels']} | Area: {info['immune_area_um2']:.2f} μm²")

    monitor_memory(inner_generate, "generate_heatmap_and_boxes")

    
# ------------------ Draw Boxes from DBSCAN Clusters ------------------
def generate_boxes_from_dbscan_clusters(points, labels, original_image, filtered_map, output_path, wsi_path, micrometer_per_pixel):
    def inner_generate():
        overlay_image = original_image.copy()
        cluster_info = []

        unique_labels = set(labels)
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue  # skip noise

            cluster_points = points[labels == cluster_id]


            # 用橢圓形擬合並放大
            ellipse = cv2.fitEllipse(cluster_points.astype(np.int32))
            (center_x, center_y), (MA, ma), angle = ellipse

            # 放大橢圓比例
            scale_factor = 1.3
            MA_enlarged = MA * scale_factor
            ma_enlarged = ma * scale_factor
            ellipse_enlarged = ((center_x, center_y), (MA_enlarged, ma_enlarged), angle)

            # 畫放大的橢圓
            cv2.ellipse(overlay_image, ellipse_enlarged, (255, 0, 0), 4)

            # 建立橢圓 mask
            mask = np.zeros_like(filtered_map, dtype=np.uint8)
            cv2.ellipse(mask, ellipse_enlarged, color=1, thickness=-1)

            # AND 運算取得橢圓內免疫像素
            immune_pixel_count = cv2.countNonZero(cv2.bitwise_and(filtered_map, filtered_map, mask=mask))
            immune_area_um2 = immune_pixel_count * (micrometer_per_pixel ** 2)

            # 取放大橢圓的外接矩形作為 cluster info 顯示
            x = int(center_x - MA_enlarged / 2)
            y = int(center_y - ma_enlarged / 2)
            w = int(MA_enlarged)
            h = int(ma_enlarged)

            # 標出免疫細胞紅點
            for (px, py) in cluster_points:
                cv2.circle(overlay_image, (int(px), int(py)), radius=2, color=(0, 0, 255), thickness=-1)

            # 儲存群聚資訊
            cluster_info.append({
                "cluster_id": cluster_id,
                "bounding_box": [x, y, w, h],
                "immune_pixels": immune_pixel_count,
                "immune_area_um2": immune_area_um2
            })

        # 儲存疊圖
        box_filename = os.path.join(output_path, f"{os.path.basename(wsi_path).split('.')[0]}_dbscan_boxes_with_points.png")
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(box_filename, overlay_image)
        print("DBSCAN Box overlay with immune cell dots saved to", box_filename)

        # 印出每群聚資訊
        print("\n===== DBSCAN Cluster Summary =====")
        for info in cluster_info:
            print(f"Cluster {info['cluster_id']} | BBox: {info['bounding_box']} | Immune Pixels: {info['immune_pixels']} | Area: {info['immune_area_um2']:.2f} μm²")

    monitor_memory(inner_generate, "generate_boxes_from_dbscan_clusters")


    
# ------------------ Draw Centers (Optional) ------------------
def generate_filtered_overlay(original_image, filtered_map, output_path, wsi_path, suffix="filtered"):
    def inner_generate():
        overlay_map = original_image.copy()
        contours, _ = cv2.findContours(filtered_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(overlay_map, (cx, cy), 3, (255, 0, 0), -1)

        overlay_output_path = os.path.join(output_path, f"{os.path.basename(wsi_path).split('.')[0]}_clust_immune_{suffix}.png")
        overlay_map = cv2.cvtColor(overlay_map, cv2.COLOR_BGR2RGB)
        cv2.imwrite(overlay_output_path, overlay_map)
        print(f"generate_{suffix}_overlay Success!")

    monitor_memory(inner_generate, "generate_filtered_overlay")

# ------------------ Main ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--immune_map", type=str, required=True, help="免疫細胞二元圖像路徑")
    parser.add_argument("--portal_map", type=str, required=True, help="Portal區域二元圖像路徑")
    parser.add_argument("--wsi_path", type=str, required=True, help="原始 mrxs 圖像路徑")
    parser.add_argument("--output_path", type=str, required=True, help="輸出的圖像存放資料夾")
    parser.add_argument("--density_thresh", type=int, default=140, help="密度圖的二值化門檻（0-255）")
    parser.add_argument("--mpp", type=float, default=0.485, help="Micrometer per pixel")
    args = parser.parse_args()

    immune_map, filtered_map = filter_binary_maps(args.immune_map, args.portal_map)
    immune_cells = extract_white_regions(filtered_map)
    points, labels = apply_dbscan_clustering(immune_cells)

    slide = openslide.OpenSlide(args.wsi_path)
    original_image = np.array(slide.get_thumbnail((filtered_map.shape[1], filtered_map.shape[0])))

    # 熱力圖和框出高密度圖
#     generate_heatmap_and_boxes(points, labels, filtered_map.shape, original_image,
#                            args.output_path, args.wsi_path, args.density_thresh, filtered_map, args.mpp)

    # 框出滿足 DBSCAN 演算法區域
    generate_boxes_from_dbscan_clusters(points, labels, original_image, filtered_map, args.output_path, args.wsi_path, args.mpp)


    # 選用：畫上免疫細胞中心點
    # generate_filtered_overlay(original_image, filtered_map, args.output_path, args.wsi_path, "filtered")
    # generate_filtered_overlay(original_image, immune_map, args.output_path, args.wsi_path, "original")

# endregion


