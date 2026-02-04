import os
import h5py
import torch
import random
import logging
import numpy as np

from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset


class listDataset(Dataset):
    def __init__(
        self,
        train,
        data_keys,
        crop_size=None,
    ):

        self.train = train
        self.data_keys = data_keys
        self.crop_size = crop_size
        
        random.shuffle(self.data_keys) if self.train else None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        # items
        rgb = self.data_keys[index]['rgb']
        tir = self.data_keys[index]['tir']
        kpoint = self.data_keys[index]['kpoint']
        fidt_map = self.data_keys[index]['fidt_map']
        img_name = self.data_keys[index]['img_name']

        # augmention
        is_aug = random.random() > 0.5
        if is_aug and self.train:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            tir = tir.transpose(Image.FLIP_LEFT_RIGHT)
            kpoint = np.ascontiguousarray(np.fliplr(kpoint))
            fidt_map = np.ascontiguousarray(np.fliplr(fidt_map))

        rgb = rgb.copy()
        tir = tir.copy()
        kpoint = kpoint.copy()
        fidt_map = fidt_map.copy()
        
        # transform
        rgb = self.transform(rgb) if self.transform else rgb
        tir = self.transform(tir) if self.transform else tir

        # crop size
        if self.train:
            fidt_map = torch.from_numpy(fidt_map).cuda()

            width = self.crop_size
            height = self.crop_size
            
            pad_y = max(0, width - rgb.shape[1])
            pad_x = max(0, height - rgb.shape[2])
            if pad_y + pad_x > 0:
                rgb = F.pad(rgb, [0, pad_x, 0, pad_y], value=0)
                tir = F.pad(tir, [0, pad_x, 0, pad_y], value=0)
                fidt_map = F.pad(fidt_map, [0, pad_x, 0, pad_y], value=0)
                kpoint = np.pad(kpoint, [(0, pad_y), (0, pad_x)], mode='constant', constant_values=0)

            crop_size_x = random.randint(0, rgb.shape[1] - width)
            crop_size_y = random.randint(0, rgb.shape[2] - height)
            rgb = rgb[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            tir = tir[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            kpoint = kpoint[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            fidt_map = fidt_map[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]

        return img_name, rgb, tir, fidt_map, kpoint


def preload_data(dataset_path, train):
    list_file = os.path.join(
        dataset_path,
        'train.npy' if train else 'test.npy',
    )
    
    logging.info(f"=> preload_data: {list_file}")
    
    with open(list_file, 'rb') as f:
            data_list = np.load(f).tolist()
            
    fidt_path = os.path.join(
        dataset_path,
        'train_fidt_map' if train else 'test_fidt_map',
    )
    
    data_keys = {}
    count = 0
    for i in range(len(data_list)):
        rgb_path, tir_path = data_list[i].split(' ')
        img_name = os.path.basename(tir_path)
        
        gt_name = img_name.replace('.jpg', '.h5')
        gt_path = os.path.join(fidt_path, gt_name)
        
        rgb = Image.open(rgb_path).convert('RGB')
        tir = Image.open(tir_path).convert('RGB')
        
        if rgb.size != tir.size:
            rgb = ImageOps.exif_transpose(rgb)
        
        # load_data_fidt
        gt = h5py.File(gt_path)
        kpoint = np.asarray(gt['kpoint'])
        fidt_map = np.asarray(gt['fidt_map'])
        
        # ignore some small resolution images
        if min(fidt_map.shape[0], fidt_map.shape[1]) < 256 and train == True:
            continue
        
        # key data
        data_key = {
            'rgb': rgb,
            'tir': tir,
            'kpoint': kpoint,
            'fidt_map': fidt_map,
            'img_name': img_name,
        }
        data_keys[count] = data_key
        count += 1

    return data_keys
