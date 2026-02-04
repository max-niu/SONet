import os
import sys
import cv2
import json
import h5py
import glob
import math
import torch

import numpy as np
import xml.etree.ElementTree as ET


def get_json_points(file_path):
    with open(file_path, 'r') as file:
        points = json.load(file)
    return points


def get_xml_points(label):
    tree = ET.parse(label)
    root = tree.getroot()
    points = [[int(obj.find('point').find('x').text), int(obj.find('point').find('y').text)]for obj in root.findall('object')]
    return points
    

def distance(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map
    
    
def dronergbt_fidt_generate(img_paths):
    for img_path in img_paths:
        print(f"[INFO]Generating: {img_path}")
        img_data = cv2.imread(img_path)

        gt_file = img_path.replace('Infrared', 'GT_').replace('.jpg', '.xml')
        points = get_xml_points(gt_file)
        gt_data = np.asarray(points)

        fidt_map = distance(img_data, gt_data, 1)

        kpoint = np.zeros((img_data.shape[0], img_data.shape[1]))
        for i in range(0, len(gt_data)):
            if int(gt_data[i][1]) < img_data.shape[0] and int(gt_data[i][0]) < img_data.shape[1]:
                kpoint[int(gt_data[i][1]), int(gt_data[i][0])] = 1
        
        h5py_file = img_path.replace('.jpg', '.h5').replace('Train/Infrared', 'train_fidt_map') if 'Train' in img_path.split('/DroneRGBT/')[-1] else img_path.replace('.jpg', '.h5').replace('Test/Infrared', 'test_fidt_map')
        with h5py.File(h5py_file, 'w') as hf:
            hf['fidt_map'] = fidt_map
            hf['kpoint'] = kpoint

        fidt_map = fidt_map / np.max(fidt_map) * 255
        fidt_map = fidt_map.astype(np.uint8)
        fidt_map = cv2.applyColorMap(fidt_map, 2)

        # for visualization
        write_file = img_path.replace('Train/Infrared', 'train_fidt_img') if 'Train' in img_path.split('/DroneRGBT/')[-1] else img_path.replace('Test/Infrared', 'test_fidt_img')
        cv2.imwrite(write_file, fidt_map)
        

def rgbtcc_fidt_generate(img_paths):
    for img_path in img_paths:
        print(f"[INFO]Generating: {img_path}")
        img_data = cv2.imread(img_path)


        gt_file = img_path.replace('_T.jpg', '_GT.json')
        points = get_json_points(gt_file)['points']
        gt_data = np.asarray(points)

        fidt_map = distance(img_data, gt_data, 1)

        kpoint = np.zeros((img_data.shape[0], img_data.shape[1]))
        for i in range(0, len(gt_data)):
            if int(gt_data[i][1]) < img_data.shape[0] and int(gt_data[i][0]) < img_data.shape[1]:
                kpoint[int(gt_data[i][1]), int(gt_data[i][0])] = 1
        
        h5py_file = img_path.replace('.jpg', '.h5').replace('train', 'train_fidt_map') if 'train' in img_path.split('/rgbtcc/')[-1] else img_path.replace('.jpg', '.h5').replace('test', 'test_fidt_map')
        with h5py.File(h5py_file, 'w') as hf:
            hf['fidt_map'] = fidt_map
            hf['kpoint'] = kpoint

        fidt_map = fidt_map / np.max(fidt_map) * 255
        fidt_map = fidt_map.astype(np.uint8)
        fidt_map = cv2.applyColorMap(fidt_map, 2)

        # visualization
        write_file = img_path.replace('train', 'train_fidt_img') if 'train' in img_path.split('/rgbtcc/')[-1] else img_path.replace('test', 'test_fidt_img')
        cv2.imwrite(write_file, fidt_map)
        

if __name__ == '__main__':
    # usage
    assert len(sys.argv) == 3, "[ERROR] usage: python preprocess_data.py <dataset_name> <dataset_path>"
    
    # check
    dataset_name = sys.argv[1]
    dataset_path = sys.argv[2]
    
    support_list = ['DroneRGBT', 'RGBTCC']
    assert dataset_name in support_list, f"[ERROR] Your dataset={dataset_name} not supported in {support_list}."
    
    # dataset
    if dataset_name == 'DroneRGBT':
        train_path = os.path.join(dataset_path, 'Train', 'Infrared')
        test_path = os.path.join(dataset_path, 'Test', 'Infrared')
        
        train_list = glob.glob(os.path.join(train_path, '*.jpg'))
        test_list = glob.glob(os.path.join(test_path, '*.jpg'))
        
        fidt_generate = dronergbt_fidt_generate
    
    elif dataset_name == 'RGBTCC':
        train_path = os.path.join(dataset_path, 'train')
        test_path = os.path.join(dataset_path, 'test')
        
        train_list = glob.glob(os.path.join(train_path, '*_T.jpg'))
        test_list = glob.glob(os.path.join(test_path, '*_T.jpg'))
        
        fidt_generate = rgbtcc_fidt_generate
 
    else:
        print(f"[ERROR] Your dataset={dataset} not supported in {support_list}.")
        sys.exit(1)
    
    path_sets = [train_path, test_path]
    all_list = train_list + test_list
    
    train_fidt_map = os.path.join(dataset_path, 'train_fidt_map')
    train_fidt_img = os.path.join(dataset_path, 'train_fidt_img')
    test_fidt_map = os.path.join(dataset_path, 'test_fidt_map')
    test_fidt_img = os.path.join(dataset_path, 'test_fidt_img')
    
    fidt_sets = [train_fidt_map, train_fidt_img, test_fidt_map, test_fidt_img]
    for fidt in fidt_sets:
        if not os.path.exists(fidt):
            os.makedirs(fidt)
            
    train_list.sort()
    test_list.sort()
    all_list.sort()
    
    # fidt_generate
    fidt_generate(all_list)
    
    # make numpy data
    train_list_file = os.path.join(dataset_path, 'train.npy')
    test_list_file = os.path.join(dataset_path, 'test.npy')
    
    rgbt_train = []
    rgbt_test = []
    if dataset_name == 'DroneRGBT':
        for tir in train_list:
            rgb = tir.replace('Infrared', 'RGB').replace('R.jpg', '.jpg')
            rgbt_train.append(f"{rgb} {tir}")
            
        for tir in test_list:
            rgb = tir.replace('Infrared', 'RGB').replace('R.jpg', '.jpg')
            rgbt_test.append(f"{rgb} {tir}")
    
    elif dataset_name == 'RGBTCC':
        for tir in train_list:
            rgb = tir.replace('_T.jpg', '_RGB.jpg')
            rgbt_train.append(f"{rgb} {tir}")
        for tir in test_list:
            rgb = tir.replace('_T.jpg', '_RGB.jpg')
            rgbt_test.append(f"{rgb} {tir}")
        
    np.save(train_list_file, rgbt_train)
    np.save(test_list_file, rgbt_test)
    
    # info
    print(f"[INFO]FIDT generate complete!")
    print(f"[INFO]FIDT map or img saved at:")
    [print(fidt) for fidt in fidt_sets]
    
    print(f"[INFO]Data list saved at:")
    [print(save) for save in [train_list_file, test_list_file]]
