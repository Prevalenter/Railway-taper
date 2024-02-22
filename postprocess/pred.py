"""
Author: Benny
Date: Nov 2019
"""

import os
import sys

import matplotlib.pyplot as plt
import torch
import numpy as np

sys.path.append('..')

import datetime
import logging
import importlib
import shutil
import argparse
from preprocess.dataloder import RailwaySensorDataset
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from pathlib import Path
from tqdm import tqdm

# from model import resnet
# sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append( '../model')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# print(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))

# best_cls = 0
# best_t = 100
# best_r = 100

best_mse = 100

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the points')

    # resnet ast_pytorch
    parser.add_argument('--model', default='ast_pytorch', help='model name [default: mmn]')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')

    parser.add_argument('--is_taper', type=int, default=0, help='shuffle the points')

    parser.add_argument('--time_str', default='2024-01-05_02-58', help='time string')

    parser.add_argument('--model_size', default='tiny224', help='model_size')
    # ft_stride
    parser.add_argument('--ft_stride', default=10, type=int, help='fstride and tstride')
    parser.add_argument('--imagenet_pretrain', default=True, type=bool, help='imagenet_pretrain')
    parser.add_argument('--audioset_pretrain', default=False, type=bool, help='imagenet_pretrain')

    # 0.0001
    parser.add_argument('--learning_rate', default=0.008, type=float, help='learning rate in training')
    parser.add_argument('--seed', type=int, default=1, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')

    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def calcu_metric(gt, pred):
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    mse = ((gt - pred) ** 2).mean()
    mae = np.abs(gt - pred).mean()
    return mse, mae


# def test(model, loader, exp_dir, criterion, log_string, cur_epoch):
def test(model, loader, exp_dir, criterion, log_string, cur_epoch, is_taper):
    # global best_cls, best_r, best_t
    global best_mse
    mean_correct = []

    classifier = model.eval()
    # mse_dict = {i: [] for i in range(2)}

    pred_list = []
    y_list = []

    for batch_id, item in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):

        # x, y = item
        x, y = item['spect'], item['gt'][:, [is_taper]]

        if not args.use_cpu:
            x, y = x.cuda().float(), y.cuda().float()

        pred = classifier(x)
        y_list.append(y[:, 0].float().cpu().numpy())
        pred_list.append(pred[:, 0].float().cpu().numpy())

    y_list = np.array(y_list)
    pred_list = np.array(pred_list)

    model.train()
    # return instance_acc
    return y_list, pred_list

def main(args):
    # def log_string(str):
    #     logger.info(str)
    #     print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # '''CREATE DIR'''
    # timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))



    exp_dir = Path(f'../log/{args.model}/{args.time_str}')
    # # exp_dir.mkdir(exist_ok=True)
    # exp_dir = exp_dir.joinpath(args.model)
    # # exp_dir.mkdir(exist_ok=True)
    # if args.log_dir is None:
    #     exp_dir = exp_dir.joinpath(timestr)
    # else:
    #     exp_dir = exp_dir.joinpath(args.log_dir)
    # exp_dir.mkdir(exist_ok=True)
    # checkpoints_dir = exp_dir.joinpath('checkpoints/')
    # checkpoints_dir.mkdir(exist_ok=True)
    # log_dir = exp_dir.joinpath('logs/')
    # log_dir.mkdir(exist_ok=True)

    # '''LOG'''
    # args = parse_args()
    # logger = logging.getLogger("Model")
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    train_dataset = RailwaySensorDataset(is_train=True)
    test_dataset = RailwaySensorDataset(is_train=False)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=10)

    '''MODEL LOADING'''
    # num_class = args.num_category
    model = importlib.import_module(args.model)
    # shutil.copy('../model/%s.py' % args.model, str(exp_dir))

    classifier = model.get_model(args=args)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    print(os.listdir(str(exp_dir)+'/checkpoints'))

    print(str(exp_dir) + '/checkpoints/best_model.pth')
    checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
    start_epoch = checkpoint['epoch']
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # try:
    #
    #     print(str(exp_dir) + '/checkpoints/best_model.pth')
    #     checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     log_string('Use pretrain model')
    # except:
    #     log_string('No existing model, starting training from scratch...')
    #     start_epoch = 0



    with torch.no_grad():
        y_train, pred_train = test(classifier.eval(), trainDataLoader, exp_dir, criterion, None, 0, args.is_taper)
        y_test, pred_test = test(classifier.eval(), testDataLoader, exp_dir, criterion, None, 0, args.is_taper)

        np.save(str(exp_dir)+'/y_test.npy', y_test)
        np.save(str(exp_dir)+'/pred_test.npy', pred_test)

        plt.scatter(y_train, pred_train, marker='s', facecolors='none', edgecolors='k', label='Train', s=80)
        plt.scatter(y_test, pred_test, marker='o', facecolors='none', edgecolors='b', label='Test', s=80)


        print('metric is: ', mean_squared_error(pred_test.flatten(), y_test.flatten()),
              pearsonr(pred_test.flatten(), y_test.flatten()))

        plt.xlabel('Ground Truth', fontsize=16)
        plt.ylabel('Prediction', fontsize=16)
        plt.subplots_adjust(top=0.9, right=0.95)
        plt.legend()
        # plt.show()
        plt.savefig(str(exp_dir) + '/rst.png', dpi=400)




if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed * 100)
    torch.cuda.manual_seed_all(args.seed * 100)
    np.random.seed(args.seed * 100)
    # print(args)
    main(args)
