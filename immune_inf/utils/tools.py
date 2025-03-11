import cv2
import json
import pyvips
import numpy as np
import torch
import scipy.ndimage as nd
import torch.nn.functional as F
import xml.dom.minidom
import xml.etree.ElementTree as ET
from tqdm import tqdm


# +
def upsample_grids(grids, size, device="cpu"):
    grids = grids.permute(0, 3, 1, 2)
    out = F.interpolate(grids, size=size, mode='bilinear', align_corners=True)
    out = out.permute(0, 2, 3, 1)
    return out


def transform_landmarks_v2(landmarks, displacement_grid):

    u_x = displacement_grid[..., 0].squeeze()
    u_y = displacement_grid[..., 1].squeeze()

    gy, gx = torch.meshgrid(torch.arange(
        displacement_grid.shape[1]), torch.arange(displacement_grid.shape[2]))
    gy = gy.type(torch.FloatTensor)
    gx = gx.type(torch.FloatTensor)

    grid_x = gx
    grid_y = gy

    u_x = -(u_x + 1)*(displacement_grid.shape[2]-1)/2 + grid_x
    u_y = -(u_y + 1)*(displacement_grid.shape[1]-1)/2 + grid_y

    landmarks_tmp = landmarks[..., :: -1].transpose(1, 0)
    landmarks = landmarks[..., :: -1].transpose(1, 0)
    ux = nd.map_coordinates(u_x, np.array(
        landmarks_tmp), order=1)
    uy = nd.map_coordinates(u_y, np.array(
        landmarks_tmp), order=1)

    return np.array([landmarks[1] - ux, landmarks[0] - uy]).transpose(1, 0)


# -

def Coordinate_trans(trans, x, y, z):
    temp = trans.copy()
    temp[:, -1] = temp[:, -1]*z

    B = np.array([[x], [y], [1]])
    C = np.dot(temp, B)

    return C[0][0], C[1][0]


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def json2xml_convert(json_path):

    annotations_json = json.load(open(json_path))["annotation"]
    root = ET.Element("ASAP_Annotations")
    annotations = ET.SubElement(root, "Annotations")
    group_list = []

    for annotation_json in annotations_json:
        annotation = ET.SubElement(annotations, "Annotation")
        annotation.set('Name', annotation_json["name"])
        annotation.set('Type', annotation_json["type"])
        annotation.set('PartOfGroup', annotation_json["partOfGroup"])
        if annotation_json["partOfGroup"] != 'None':
            if [annotation_json["partOfGroup"], annotation_json["color"]] not in group_list:
                group_list.append(
                    [annotation_json["partOfGroup"], annotation_json["color"]])
        annotation.set('Color', annotation_json["color"])
        coordinates_json = annotation_json["coordinates"]
        coordinates = ET.SubElement(annotation, "Coordinates")
        order = 0
        for coordinate_json in coordinates_json:
            x = float(coordinate_json["x"])
            y = float(coordinate_json["y"])           
            coordinate = ET.SubElement(coordinates, "Coordinate")
            coordinate.set("Order", str(order))
            coordinate.set("X", str(x))
            coordinate.set("Y", str(y))
            order += 1

    annotationgroups = ET.SubElement(root, "AnnotationGroups")
    for temp in group_list:
        group = ET.SubElement(annotationgroups, "Group")
        group.set('Name', temp[0])
        group.set('PartOfGroup', 'None')
        group.set('Color', temp[1])
        
    return root


def write_xml(root, output_path=r'_test.xml'):
    xmlstr = ET.tostring(root).decode()
    xmlstr = xml.dom.minidom.parseString(xmlstr)
    xmlstr = xmlstr.toprettyxml()
    f = open(output_path, 'w')
    f.writelines(xmlstr)
    f.close()


def xml2json_convert(inputxml):

    xml_root = ET.parse(inputxml)
    annotations = xml_root.findall(".//Annotation")
    this_annotation = []
    for annotation in annotations:
        this_coordinate = []
        for coordinate in annotation.iter('Coordinate'):
            this_coordinate.append(
                {"y": coordinate.attrib['Y'], "x": coordinate.attrib['X']})
        this_annotation.append(
            {"name": annotation.attrib["Name"], "type": annotation.attrib["Type"], "partOfGroup": annotation.attrib["PartOfGroup"], "color": annotation.attrib["Color"], "coordinates": this_coordinate})
    new_data = {"annotation": this_annotation}

    return new_data


def write_json(json_data, output_path=r'_test.json'):
    with open(json_data, 'w', newline='') as jsonfile:
        json.dump(json_data, jsonfile)


class WSI_Reader():

    def __init__(self, slide_path, mag=None):

        self.slide_path = slide_path
        self.slide = None
        self.mag = mag
        self.max_level = None
        self.mask = np.array([None])
        self.mag_dict = dict()
        self.print_flag = False
        self.init_set()

    def init_set(self):
        self.set_mag_dict()        

    def set_slide(self, level=0):
        try:
            self.slide = pyvips.Image.new_from_file(
                self.slide_path, level=level)
        except:
            self.slide = pyvips.Image.new_from_file(
                self.slide_path, page=level)

    def set_mask(self, mask):
        self.mask = mask

    def set_mag_dict(self):
        self.set_mag()
        if self.print_flag:
            print(f"slide: {self.slide_path}, mag: {self.mag}")
        self.max_level = int(self.slide.get('openslide.level-count'))-1
        width, _ = self.get_wh(0)
        current_mag = self.mag
        wsi_level = 0

        while (wsi_level <= self.max_level):
            if round(width/(self.get_wh(wsi_level))[0]) == 1:
                self.mag_dict.setdefault(wsi_level, current_mag)
                wsi_level += 1
            current_mag *= 0.5
            width *= 0.5

    def set_mag(self):
        self.set_slide(0)
        self.mag = round(float(self.slide.get('xres'))/100, -1)

    def get_wh(self, level):
        height = int(self.slide.get(
            'openslide.level[{}].height'.format(level)))
        width = int(self.slide.get(
            'openslide.level[{}].width'.format(level)))
        return int(width), int(height)

    def get_mag_dict(self):
        return self.mag_dict

    def read_wsi(self, mag):
        level = 0
        while (self.mag_dict[level] >= mag):
            level += 1
            if (level > self.max_level):
                break
        level -= 1

        self.set_slide(level)
        w, h = self.slide.width, self.slide.height
        image = pyvips.Region.new(self.slide)
        size = 512
        rescale = 1

        if mag < self.mag_dict[level]:
            new_w, new_h = int(
                mag/self.mag_dict[level]*w), int(mag/self.mag_dict[level]*h)
            rescale = int(self.mag_dict[level]/mag)
        else:
            new_w, new_h = w, h
        # new_w, new_h = w, h

        wsi_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        wsi_img += 255

        if self.mask.any() != None:
            mask = cv2.resize(self.mask, (new_w, new_h),
                              interpolation=cv2.INTER_AREA)
            mask[mask < 128] = 0

        for x in tqdm(range(0, int(new_w), int(size)), desc='Loading WSI'):
            for y in range(0, int(new_h), int(size)):
                if self.mask.any() != None:
                    if (mask[y:y+size, x:x+size].mean() == 0):
                        continue
                patch_w, patch_h = size, size
                if x + size > new_w:
                    patch_w = new_w - x
                if y + size > new_h:
                    patch_h = new_h - y
                patch_data = image.fetch(
                    x*rescale, y*rescale, patch_w*rescale, patch_h*rescale)
                channel = int(len(patch_data) /
                              ((patch_w*rescale)*(patch_h*rescale)))
                patch = np.ndarray(buffer=patch_data, dtype=np.uint8, shape=[
                    patch_h*rescale, patch_w*rescale, channel])
                patch = rgba2rgb(patch)
                if rescale != 1:
                    patch = cv2.resize(patch, (patch_w, patch_h))
                if self.mask.any() != None:
                    patch[mask[y:y+patch_h, x:x+patch_w] == 0] = [255, 255, 255]
                wsi_img[y:y+patch_h, x:x+patch_w] = patch

        return wsi_img

#Example
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    wsi_path = '/work/jiang2020/Core-Needle Biopsy/21033546/HE.mrxs'
    example_reader = WSI_Reader(wsi_path)
    wsi_img = example_reader.read_wsi(0.625)
    plt.imshow(wsi_img),plt.show()


