import os
import torch
import logging
import warnings
import numpy as np
import torch.nn as nn

from args import args
from dataset import preload_data, listDataset
from utils import set_seed, set_test_logger, eval_game, LMDS_counting, eval_relative

from models.rgbtcc import get_model

warnings.filterwarnings('ignore')


def main(args):
    test_data = preload_data(args['dataset_path'], train=False)

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
    
    torch.set_num_threads(args['workers'])

    # inference
    validate(test_data, model, args)

def validate(data_keys, model, args):
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        listDataset(
            train=False,
            data_keys=data_keys,
        ),
        batch_size=1,
    )
    
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0
    
    model.eval()
    for i, (fname, rgb, tir, fidt_map, kpoint) in enumerate(test_loader):
        rgb = rgb.cuda()
        tir = tir.cuda()
        fidt_map = fidt_map.unsqueeze(0)

        with torch.no_grad():
            d6 = model([rgb, tir])

            # return counting and coordinates
            count, pred_kpoint = LMDS_counting(d6)
            
            # save density_map
            save_file = os.path.join(
                args_dict['save_path'],
                fname[0].replace('R.jpg', '_RGB.pt').replace('_T.jpg', '_RGB.pt'),
            )
            
            torch.save(d6, save_file)
            logging.info(f"=> save map: {save_file}")
            
            for L in range(4):
                abs_error, square_error = eval_game(pred_kpoint, kpoint, L)
                game[L] += abs_error
                mse[L] += square_error
                relative_error = eval_relative(pred_kpoint, kpoint)
                total_relative_error += relative_error
            
    N = len(test_loader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N
    
    logging.info(f"=> test: num={N}, GAME0={game[0]:0.2f}, GAME1={game[1]:0.2f}, GAME2={game[2]:0.2f}, GAME3={game[3]:0.2f}, MSE={mse[0].cpu().numpy():0.2f}")


if __name__ == '__main__':
    args_dict = vars(args)
    set_test_logger(args_dict)
    set_seed(args_dict['seed'])

    main(args_dict)