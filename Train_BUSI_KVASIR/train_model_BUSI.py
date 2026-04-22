
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch.backends import cudnn
from Load_Dataset_BUSI import RandomGenerator, ValGenerator, ImageToImage2D

from torch.utils.data import DataLoader
import logging
from Train_one_epoch_BUSI import train_one_epoch
import Config_BUSI as config
from torchvision import transforms
from utils import WeightedDiceBCE,CosineAnnealingWarmRestarts
from thop import profile



def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr
import torch.nn.functional as F
def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):

    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])

    if config.task_name == 'BUSI':
        # train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        # val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_tf, image_size=config.img_size)

    elif config.task_name == 'Kvasir':
        # train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        # val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_tf, image_size=config.img_size)


    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)

    lr = config.learning_rate
    logger.info(model_type)

    if model_type == 'AmGANet':
        from model.AmGANet_2D import AmGANet_2D
        import argparse
        from model.config import get_config
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str,
                            metavar="FILE",
                            default="model/configs/AmGANet.yaml",
                            help='path to config file')
        parser.add_argument('--resume', help='resume from checkpoint')
        args1 = parser.parse_args()
        configs = get_config(args1.cfg)

        model = AmGANet_2D(configs, task_name=config.task_name, num_classes=1)
        model.load_from(configs)

    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    # lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)


    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: {}'.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1

    # ===== 总训练开始时间 =====
    total_start_time = time.time()

    for epoch in range(config.epochs):
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)

        # ===== 单个 epoch 开始时间 =====
        epoch_start_time = time.time()

        # ===== train 开始 =====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_start_time = time.time()

        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time = time.time() - train_start_time
        logger.info('Train time for epoch {}: {} ({:.2f}s)'.format(
            epoch + 1, format_seconds(train_time), train_time))

        # ===== val 开始 =====
        logger.info('Validation')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        val_start_time = time.time()

        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(
                val_loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        val_time = time.time() - val_start_time
        logger.info('Val time for epoch {}: {} ({:.2f}s)'.format(
            epoch + 1, format_seconds(val_time), val_time))

        # ===== epoch 总时间 =====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start_time
        logger.info('Total epoch {} time: {} ({:.2f}s)'.format(
            epoch + 1, format_seconds(epoch_time), epoch_time))

        # TensorBoard 里记录时间
        if writer is not None:
            writer.add_scalar('time/train_epoch_sec', train_time, epoch + 1)
            writer.add_scalar('time/val_epoch_sec', val_time, epoch + 1)
            writer.add_scalar('time/epoch_total_sec', epoch_time, epoch + 1)

        # Save best model
        if val_dice > max_dice:
            if epoch + 1 > 5:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch,
                    'best_model': True,
                    'model': model_type,
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'optimizer': optimizer.state_dict()
                }, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))

        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(
            early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    # ===== 总训练结束时间 =====
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - total_start_time
    logger.info('\n================ Training Finished ================')
    logger.info('Total training time: {} ({:.2f}s)'.format(
        format_seconds(total_time), total_time))

    if writer is not None:
        writer.add_text('time/summary', 'Total training time: {}'.format(format_seconds(total_time)))
        writer.close()

    return model

if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)
