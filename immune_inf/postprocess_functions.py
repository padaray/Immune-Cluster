import cv2
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from scipy import ndimage as ndi
from skimage.segmentation import watershed as wts


# Dilate image
def dilate(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    return dilated_image

# Erode image
def erode(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)

    return eroded_image

def filter_mask_2_contour(mask, offsets=(0, 0)):
    contours = []
    label_map = ndi.label(mask)[0]
    num_label = np.max(label_map)

    for n in range(1, num_label + 1):
        cnts, _ = cv2.findContours((label_map == n).astype(np.uint8),
                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours.append(cnts[0] + offsets)

    return contours


def watershed(img, mask, markers):
    # ref: https://docs.scipy.org/doc/scipy/
    # reference/generated/scipy.ndimage.label.html
    markers = ndi.label(markers)[0]  # create labels for each white area in mask
    labels = wts(-img, markers, mask=np.round(mask), watershed_line=True)
    mask[labels == 0] = 0

    return mask


def create_point_annotation(show_range:tuple, contours:list, x:int, y:int, filter_pixel:int):  
    """
    Create point annotation from contour points by finding the gravity center
    
    Args:
        show_range(int of tuple): size of patch
        contours(list of float): all the countours
        x (int): x in top left point of patch
        y (int): y in top left point of patch
    
    Return:
        coord_list (list of dict): list of dictionary with 
                                    point & contour annotation 
    """

    coord_list = []
    
    for cnt in contours:
        # Contour area lower than setting pixels is filtered 
        if cv2.contourArea(cnt) < filter_pixel:
            continue

        M = cv2.moments(cnt)
        # if it is a zero area, contour point be tha annotaion 
        if M['m00'] == 0:
            cx = cnt[0][0][0]
            cy = cnt[0][0][1]

        # else calculate the center of gravity 
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        # make sure the annotation within patch
        if cx <= show_range[1] and cy <= show_range[0]:
            # calculate the absolute coordinate ni WSI
            cx = cx + x
            cy = cy + y
            contour_2_save = [[c[0][0] + x, c[0][1] + y] for c in cnt.tolist()]
            Coordinates = dict(contour=contour_2_save, x = int(cx), y = int(cy))
            coord_list.append(Coordinates)

    return coord_list


def remove_cell_in_portal(all_cells_mask, portal_pred, portal_contours):
    if portal_contours != 0: 
        label_map = ndi.label(all_cells_mask)[0]
        num_label = np.max(label_map)
        cells_wo_portal_mask = np.zeros(all_cells_mask.shape, dtype=np.uint8)

        for n in range(1, num_label + 1):
            one_cell_label = np.zeros_like(label_map, dtype=np.uint8)
            one_cell_label[label_map == n] = 255
            if np.sum(one_cell_label & portal_pred) == 0:
                cells_wo_portal_mask = cells_wo_portal_mask | one_cell_label

        return cells_wo_portal_mask
    
    else:
        return all_cells_mask
