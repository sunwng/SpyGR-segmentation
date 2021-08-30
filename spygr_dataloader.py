import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

def data_list(root, quality, mode):
    # mode : train / val / test
    if quality == "fine":
        img_dir_name = "leftImg8bit_trainvaltest"
        mask_path = os.path.join(root, "gtFine_trainvaltest", "gtFine", mode)
        mask_suffix = "_gtFine_labelIds.png"
        img_suffix = "_leftImg8bit.png"
    img_path = os.path.join(root, img_dir_name, "leftImg8bit", mode)

    items = []
    categories = os.listdir(img_path)

    for c_idx in categories:
        c_items = [name.split(img_suffix)[0] for name in os.listdir(os.path.join(img_path, c_idx))]
        for item_idx in c_items:
            item = (os.path.join(img_path, c_idx, item_idx+img_suffix), os.path.join(mask_path, c_idx, item_idx+mask_suffix))
            items.append(item)

    return items

class CityScapesData(Dataset):
    def __init__(self, root_dir, quality, mode, transform, crop_size):
        self.quality = quality
        self.mode = mode
        self.transform = transform
        self.crop_size = crop_size
        self.img_list = data_list(root_dir, quality, mode)
        self.ignore_label = 255
        self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: self.ignore_label, 2: self.ignore_label,
                              3: self.ignore_label, 4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label,
                              7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label, 11: 2, 12: 3, 13: 4,
                              14: self.ignore_label, 15: self.ignore_label, 16: self.ignore_label, 17: 5,
                              18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: self.ignore_label, 30: self.ignore_label, 31: 16, 32: 17, 33: 18}
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

    def __len__(self):
        return len(self.img_list)

    def random_crop(self, img, mask):
        crop_tf = T.RandomCrop(self.crop_size)
        img = crop_tf(img)
        mask = crop_tf(mask)
        # i, j, h, w = T.RandomCrop.get_params(img, output_size=self.crop_size)
        # img = T.RandomCrop(img, i, j, h, w)
        # mask = T.RandomCrop(mask, i, j, h, w)
        return img, mask

    def __getitem__(self, index):
        img_path, mask_path = self.img_list[index]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        img, mask = self.random_crop(img, mask)

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask==k] = v
        # label = Image.fromarray(label_copy.astype(np.uint8))
        # label = np.array(label, dtype=np.float32)
        # label = torch.from_numpy(np.array(label_copy, dtype=np.int32)).long()

        if self.transform is not None:
            img = self.transform(img)
        
        mask = torch.from_numpy(np.array(mask_copy, dtype=np.int32)).long()
        
        return img, mask