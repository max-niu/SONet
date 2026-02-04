import h5py
import torch
import shutil
import numpy as np
import cv2
import os
import random
import logging
import torch.nn as nn
import math
from datetime import datetime


def eval_relative(output, target):
    output_num = output.sum()
    target_num = target.sum().float()
    relative_error = abs(output_num-target_num)/target_num
    return relative_error
    
    
def eval_game(output, target, L=0):
    # output = output[0][0].cpu().numpy()
    target = target[0]
    H, W = target.shape
    ratio = H / output.shape[0]
    output = cv2.resize(output, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio*ratio)

    assert output.shape == target.shape

    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    for i in range(p):
        for j in range(p):
            output_block = output[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            target_block = target[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]

            abs_error += abs(output_block.sum()-target_block.sum().float())
            square_error += (output_block.sum()-target_block.sum().float()).pow(2)

    return abs_error, square_error
    
    
def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
    density_map[density_map < 0] = 0

    gt_data = 255 * gt_data / np.max(gt_data)
    gt_data = gt_data[0][0]
    gt_data = gt_data.astype(np.uint8)
    gt_data = cv2.applyColorMap(gt_data, 2)

    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)

    result_img = np.hstack((gt_data, density_map))

    cv2.imwrite(os.path.join('.', output_dir, fname).replace('.jpg', '.jpg'), result_img)


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, visi, is_best, save_path, filename='checkpoint.pth'):
    torch.save(state, './' + str(save_path) + '/' + filename)
    if is_best:
        shutil.copyfile('./' + str(save_path) + '/' + filename, './' + str(save_path) + '/' + 'model_best.pth')

    for i in range(len(visi)):
        img = visi[i][0]
        output = visi[i][1]
        target = visi[i][2]
        fname = visi[i][3]
        save_results(img, target, output, str(save_path), fname[0])


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 输入固定情况下用true


def set_logger(args_dict):
    sub_dir = os.path.join(
        args_dict['save_path'],
        datetime.strftime(datetime.now(), '%m%d-%H%M%S'),
    )
    args_dict['save_path'] = sub_dir
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter(
        "[%(asctime)s %(message)s",
        "%m-%d %H:%M:%S]"
    )
    
    loger_path = os.path.join(sub_dir, 'train.log')
    fileHandler = logging.FileHandler(loger_path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
    logging.info(f"{'*'*10} train infomation {'*'*10}")
    [logging.info(f"{k}: {v}") for k, v in args_dict.items()]
    logging.info(f"{'*'*39}")
    

def set_test_logger(args_dict):
    sub_dir = args_dict['save_path']
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter(
        "[%(asctime)s %(message)s",
        "%m-%d %H:%M:%S]"
    )
    
    loger_path = os.path.join(sub_dir, 'test.log')
    fileHandler = logging.FileHandler(loger_path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
    logging.info(f"{'*'*10} test infomation {'*'*10}")
    [logging.info(f"{k}: {v}") for k, v in args_dict.items()]
    logging.info(f"{'*'*39}")
    
    
def LMDS_counting(input):
    input_max = torch.max(input).item()

    ''' find local maxima'''
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    '''set the pixel valur of local maxima as 1 for counting'''
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1

    ''' negative sample'''
    if input_max < 0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    return count, kpoint


def generate_point_map(kpoint, rate=1):
    '''obtain the location coordinates'''
    pred_coor = np.nonzero(kpoint)

    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)
    return point_map


def generate_bounding_boxes(kpoint, fname):
    '''change the data path'''
    Img_data = cv2.imread(
        '/home/dkliang/projects/synchronous/datasets/ShanghaiTech/part_A_final/test_data/images/' + fname[0])
    ori_Img_data = Img_data.copy()

    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    distances, locations = tree.query(pts, k=4)
    for index, pt in enumerate(pts):
        pt2d = np.zeros(kpoint.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if np.sum(kpoint) > 1:
            sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
        else:
            sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
        sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)

        if sigma < 6:
            t = 2
        else:
            t = 2
        Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                 (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)

    return ori_Img_data, Img_data


def show_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
