import os
import time
import math
import torch
import shutil
import logging
import warnings
import torch.nn as nn

from args import args
from dataset import preload_data, listDataset
from utils import set_seed, set_logger, generate_point_map, AverageMeter, LMDS_counting

from models.rgbtcc import get_model


warnings.filterwarnings('ignore')


def train(data_keys, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        listDataset(
            train=True,
            data_keys=data_keys,
            crop_size=args['crop_size'],
        ),
        batch_size=args['batch_size'],
        drop_last=False,
    )
    args['lr'] = optimizer.param_groups[0]['lr']

    model.train()
    end = time.time()
    for i, (fname, rgb, tir, fidt_map, kpoint) in enumerate(train_loader):

        data_time.update(time.time() - end)
        rgb = rgb.cuda()
        tir = tir.cuda()
        fidt_map = fidt_map.type(torch.FloatTensor).unsqueeze(1).cuda()

        d6 = model([rgb, tir])

        if d6.shape != fidt_map.shape:
            print("the shape is wrong, please check. Both of prediction and GT should be [B, C, H, W].")
            exit()
            
        loss = criterion(d6, fidt_map)
        losses.update(loss.item(), rgb.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0 or i == len(train_loader) - 1:
            logging.info(
                f"Train: "
                f"Epoch==({epoch:4}/{args['epochs']:4}), "
                f"Step==({i:3}/{len(train_loader):3}), "
                f"DataTime=={data_time.val:.3f}, "
                f"BatchTime=={batch_time.avg:.3f}, "
                f"LR=={args['lr']:.6f}, "
                f"Loss=={losses.avg:.4f}\t"
            )


def validate(data_keys, model, args):
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        listDataset(
            train=False,
            data_keys=data_keys,
        ),
        batch_size=1,
    )
    
    mae = 0.0
    mse = 0.0
    index = 0
    model.eval()
    
    for i, (fname, rgb, tir, fidt_map, kpoint) in enumerate(test_loader):
        rgb = rgb.cuda()
        tir = tir.cuda()
        fidt_map = fidt_map.unsqueeze(0)

        with torch.no_grad():
            d6 = model([rgb, tir])

            # return counting and coordinates
            count, pred_kpoint = LMDS_counting(d6)
            point_map = generate_point_map(pred_kpoint)

            if args['visual'] == True:
                if not os.path.exists(args['save_path'] + '_box/'):
                    os.makedirs(args['save_path'] + '_box/')
                ori_img, box_img = generate_bounding_boxes(pred_kpoint, fname)
                show_fidt = show_map(d6.data.cpu().numpy())
                gt_show = show_map(fidt_map.data.cpu().numpy())
                res = np.hstack((ori_img, gt_show, show_fidt, point_map, box_img))
                cv2.imwrite(args['save_path'] + '_box/' + fname[0], res)

        gt_count = torch.sum(kpoint).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    return mae, mse


def main(args):
    # data
    train_data = preload_data(args['dataset_path'], train=True)
    test_data = preload_data(args['dataset_path'], train=False)

    # model
    model = get_model()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    
    pretrain = args['pretrain']
    if pretrain:
        if os.path.isfile(pretrain):
            logging.info(f"=> loading pretrained checkpoint: {pretrain}")
            checkpoint = torch.load(pretrain)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_pred']
            logging.info(f"=> pretrained checkpoint: start_epoch=={args['start_epoch']}, best_pred={args['best_pred']}")
        else:
            logging.info(f"not found checkpoint: {pretrain}")

    # optimizer
    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ], 
        lr=args['lr'],
        weight_decay=args['weight_decay'],
    )
    
    # criterion
    criterion = nn.MSELoss(size_average=False).cuda()

    # train
    torch.set_num_threads(args['workers'])
    for epoch in range(args['start_epoch'], args['epochs']):
        start = time.time()
        train(train_data, model, criterion, optimizer, epoch, args)
        end1 = time.time()

        # validate
        is_best = False
        if epoch >= 0:
            mae, mse = validate(test_data, model, args)
            end2 = time.time()

            is_best = mae < args['best_pred']
            args['best_pred'] = min(mae, args['best_pred'])

            logging.info(
                f"Valid: "
                f"Epoch==({epoch:4}/{args['epochs']:4}), "
                f"MAE=={mae:.3f}, "
                f"MSE=={mse:.3f}, "
                f"BEST *MAE=={args['best_pred']:.3f}\t"
            )
        
        # save
        save_dict = {
            'epoch': epoch,
            'best_pred': args['best_pred'],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        torch.save(
            save_dict,
            os.path.join(args['save_path'], 'checkpoint.pth'),
        )
        
        if is_best:
            shutil.copyfile(
                os.path.join(args['save_path'], 'checkpoint.pth'),
                os.path.join(args['save_path'], f'model_best_{epoch}.pth'),
            )
        

if __name__ == '__main__':
    args_dict = vars(args)
    set_logger(args_dict)
    set_seed(args_dict['seed'])
    
    main(args_dict)
