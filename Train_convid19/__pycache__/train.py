import torch
from torch.utils.data import DataLoader
# from utils.dataset import QaTa
from utils.dataset1 import QaTa
import utils.config as config
from torch.optim import lr_scheduler
from engine.wrapper1 import LanGuideMedSegWrapper1

import pytorch_lightning as pl    
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import logging
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

# default='LanGuideMedSeg-MICCAI2023-main/config/training.yaml',

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')
    # parser.add_argument('--cfg', type=str,
    #                     metavar="FILE",
    #                     default="LanGuideMedSeg-MICCAI2023-main/utils/ours/configs/agileFormer_tiny.yaml",
    #                     help='path to config file', )
    
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg

from thop import profile		 ## 导入thop模块
def cal_params_flops(model, size, logger):
    model = model.cuda()
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')

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

if __name__ == '__main__':

    args = get_parser()
    print("cuda:",torch.cuda.is_available())
    ds_train = QaTa(dataname="QaTa",
                    csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    image_size=args.image_size,
                    mode='train')

    ds_valid = QaTa(dataname="QaTa",
                    csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    image_size=args.image_size,
                    mode='valid')

    # ds_train = QaTa(dataname="MosMedData",
    #                 csv_path=args.train_csv_path,
    #                 root_path=args.train_root_path,
    #                 image_size=args.image_size,
    #                 mode='train')

    # ds_valid = QaTa(dataname="MosMedData",
    #                 csv_path=args.val_csv_path,
    #                 root_path=args.val_root_path,
    #                 image_size=args.image_size,
    #                 mode='valid')
    
    # ds_train = QaTa(dataname="MoNuSeg",
    #                 csv_path=args.train_csv_path,
    #                 root_path=args.train_root_path,
    #                 image_size=args.image_size,
    #                 mode='train')

    # ds_valid = QaTa(dataname="MoNuSeg",
    #                 csv_path=args.val_csv_path,
    #                 root_path=args.val_root_path,
    #                 image_size=args.image_size,
    #                 mode='valid')

    # ds_train = QaTa(dataname="BUSI",
    #                 root_path=args.train_root_path,
    #                 image_size=args.image_size,
    #                 mode='train')

    # ds_valid = QaTa(dataname="BUSI",
    #                 root_path=args.val_root_path,
    #                 image_size=args.image_size,
    #                 mode='valid')

    # dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size)
    # dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size)
    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)
    model = LanGuideMedSegWrapper1(args)

    ## 1. setting recall function
    model_ckpt = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor='val_loss',
        # monitor='val_MIoU',
        save_top_k=1,
        mode='min',
        verbose=True,
    )

    early_stopping = EarlyStopping(monitor = 'val_loss',
                            patience=args.patience,
                            mode = 'min'
    )
    # logger_path = args.model_save_path + ".log"
    # logger = logger_config(log_path=logger_path)

    # cal_params_flops(model, 224, logger)
    ## 2. setting trainer

    trainer = pl.Trainer(logger=True,
                        min_epochs=args.min_epochs,max_epochs=args.max_epochs,
                        accelerator='gpu', 
                        devices=args.device,
                        callbacks=[model_ckpt,early_stopping],
                        enable_progress_bar=True,
                        ) 

    ## 3. start training
    print('start training')
    trainer.fit(model,dl_train,dl_valid)
    print('done training')

