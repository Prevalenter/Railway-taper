"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

sys.path.append('..')

import datetime
import logging
import importlib
import shutil
import argparse
from preprocess.dataloder import RailwaySensorDataset

from pathlib import Path
from tqdm import tqdm

# from model import resnet
# sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append( '../model')

BASE_DIR = os.path.dirname( os.path.abspath(__file__) )
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
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')

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


def test(model, loader, exp_dir, criterion, log_string, cur_epoch):
    # global best_cls, best_r, best_t
    global best_mse
    mean_correct = []

    classifier = model.eval()
    mse_dict = {i: [] for i in range(2)}

    for batch_id, item in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):

        x, y = item

        if not args.use_cpu:
            x, y = x.cuda().float(), y.cuda().float()

        pred = classifier(x)


        # print(y.shape, pred.shape)
        # r, t
        loss = criterion(y, pred)

        for i in range(2):
            mse_dict[i].append( criterion(y[:, i], pred[:, i]).cuda().float().cpu() )

    mse_cur = []
    for i in range(2):
        mse_cur.append(np.array(mse_dict[i]).mean())
        print( i, mse_cur[-1] )

    mse_cur = np.array(mse_cur)

    # best_cls = np.max([cls_mean.mean(), best_cls])
    best_mse = np.min([mse_cur.mean(), best_mse])


    # log_string(f'Cls correct [{cls_mean.mean()}, {best_cls}] {cls_mean}')
    log_string(f'MSE: [{mse_cur.mean()}/ {best_mse}] {mse_cur}')

    # instance_acc = np.mean(mean_correct)
    # instance_acc = rmse_r_mean.mean()

    model.train()
    # return instance_acc
    return mse_cur.mean()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('../log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.model)
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    # train_dataset = MechaPartMatchLoader(args=args, partition='train', num_nodes=args.num_kp)
    # test_dataset = MechaPartMatchLoader(args=args, partition='test', num_nodes=args.num_kp)
    train_dataset = RailwaySensorDataset(is_train=True)
    test_dataset = RailwaySensorDataset(is_train=False)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    # num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('../model/%s.py' % args.model, str(exp_dir))

    classifier = model.get_model(args=args)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    
    global_epoch = 0
    global_step = 0
    best_instance_acc = 1000.0
    best_class_acc = 0.0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()


        for batch_id, item in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):

            optimizer.zero_grad()
            x, y = item

            if not args.use_cpu:
                x, y = x.cuda().float(), y.cuda().float()

            pred = classifier(x)

            loss = criterion(y.float(), pred.float())

            mean_correct.append(loss.item())
            loss.backward()
            optimizer.step()
            # scheduler.step()
            global_step += 1
            

            # break

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc = test(classifier.eval(), testDataLoader, exp_dir, criterion, log_string, global_epoch)
        #     best_instance_acc
            if (best_instance_acc >= instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

                # print('-'*50)
                # print(y)
                # print(pred)
                # print('-'*50)


        #     log_string('Test Instance Accuracy: %f' % (instance_acc))
        #     log_string('Best Instance Accuracy: %f' % (best_instance_acc))
        #
            # if (best_instance_acc >= instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed * 100)
    torch.cuda.manual_seed_all(args.seed * 100)
    np.random.seed(args.seed * 100)
    # print(args)
    main(args)
