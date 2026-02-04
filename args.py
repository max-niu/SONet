import argparse

parser = argparse.ArgumentParser(description='FIDTM')

# cfg
parser.add_argument('--dataset_path', type=str, required=True,
                    help='choice train dataset')
parser.add_argument('--save_path', type=str, required=True,
                    help='save checkpoint directory')

# train                    
parser.add_argument('--workers', type=int, default=16,
                    help='load data workers')
parser.add_argument('--print_freq', type=int, default=20,
                    help='print frequency')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')
parser.add_argument('--epochs', type=int, default=500,
                    help='end epoch for training')
parser.add_argument('--pretrain', type=str, default=None,
                    help='pretrained model directory')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')
parser.add_argument('--visual', type=bool, default=False,
                    help='visual for bounding box. ')          

# dataset
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--crop_size', type=int, default=256,
                    help='crop size for training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# optimizer
parser.add_argument('--lr', type=float, default= 1e-4,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')

args = parser.parse_args()
