import torch
import numpy as np

def IOU(mask, output):
    # print(mask.shape)
    # print(output.shape)

    output = torch.argmax(output, dim=1)
    intersection = torch.sum(torch.mul(mask, output))
    union = torch.sum(mask) + torch.sum(output) - intersection
    
    # print("intersection", intersection)
    # print("union", union)
    # print(output)

    return intersection / union

def Precision(mask, output):
    output = torch.round(output)
    intersection = torch.sum(torch.mul(mask, output))
    return intersection / torch.sum(output)

def Recall(mask, output):
    output = torch.round(output)
    intersection = torch.sum(torch.mul(mask, output))
    return intersection / torch.sum(mask)

def Dice_cofficient(mask, output):
    output = torch.round(output)
    numerator = torch.sum(torch.mul(mask, output))
    denominator = torch.sum(mask ) + torch.sum(output)
    return numerator * 2 / denominator

class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)