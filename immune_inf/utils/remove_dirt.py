import cv2
import torch
import numpy as np
from tqdm import tqdm
from utils.unet_model import UNet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class custom_data_set(Dataset):

    def __init__(self, wsi_img, size=512):
        (w, h) = wsi_img.shape[1], wsi_img.shape[0]
        wsi_img = cv2.copyMakeBorder(
            wsi_img, int(size/2), int(size/2), int(size/2), int(size/2), borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        self.patch_pointer = []
        self.patchs = []
        for x in tqdm(range(0, int(w), int(size/2)), desc='Loading Data'):
            for y in range(0, int(h), int(size/2)):
                if wsi_img[y+int(size/4):y+int(3*size/4), x+int(size/4):x+int(3*size/4)].mean() < 255:
                    self.patch_pointer.append({'x': x, 'y': y, 'size': size})
                    self.patchs.append(wsi_img[y:y+size, x:x+size])

    def __len__(self):
        return len(self.patch_pointer)

    def __getitem__(self, idx):

        pointer = self.patch_pointer[idx]
        img = self.patchs[idx]
        img = img / 255
        img = img.transpose((2, 0, 1))

        return {
            'image':  torch.as_tensor(img.copy()).float().contiguous(),
            'pointer': pointer
        }


def run(net,
        device,
        wsi_img,
        batch_size: int = 16,
        patch_size: int = 512):

    dataset = custom_data_set(wsi_img, patch_size)
    stride = patch_size/2
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=False, **loader_args)

    net.eval()
    result = np.zeros_like(wsi_img[:, :, 0])
    (w, h) = result.shape[1], result.shape[0]
    result = cv2.copyMakeBorder(result, int(stride), int(stride), int(stride), int(
        stride), borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    with tqdm(total=len(dataset), desc='Progress', unit='img') as pbar:
        for batch in train_loader:
            images = batch['image']
            images = images.to(device=device, dtype=torch.float32)

            pointer = batch['pointer']
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    masks_pred = net(images)

            batch_x = pointer['x'].cpu().numpy()
            batch_y = pointer['y'].cpu().numpy()

            for i in range(len(batch_x)):
                x = batch_x[i]
                y = batch_y[i]
                patch = torch.softmax(masks_pred, dim=1).argmax(dim=1)[
                    i].float().cpu().numpy()
                patch[patch > 0.5] = 255
                patch[patch <= 0.5] = 0
                result[y+int(stride/2):y+int(3*stride/2), x+int(stride/2):x
                       + int(3*stride/2)] = patch[int(stride/2):int(3*stride/2), int(stride/2):int(3*stride/2)]

            pbar.update(images.shape[0])
    result = result[int(stride):int(stride)+h, int(stride):int(stride)+w]
    return result


def remove_dirt(wsi_img):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = "./immune_inf/utils/unet_seg_new_weight_221122.pth"
    batch_size = 16
    patch_size = 512

    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    net.load_state_dict(torch.load(model, map_location=device))
    net.to(device=device)

    mask = run(net=net,
               wsi_img=wsi_img,
               device=device,
               batch_size=batch_size,
               patch_size=patch_size)

    return mask
