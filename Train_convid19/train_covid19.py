import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
from torch.utils.data import DataLoader
from Load_Dataset_covid import QaTa
import config_Covid19 as config
from wrapper import AmGANet_2DWrapper
import pytorch_lightning as pl    
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import logging
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import time


def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='/home/y1408/CAI/third_paper/AmGANet/Train_convid19/config/training.yaml',
                        type=str,
                        help='config file')
    
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg

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
    
    if args.task_name == "Covid19_X":

        ds_train = QaTa(dataname="Covid19_X",
                        csv_path=args.train_csv_path,
                        root_path=args.train_root_path,
                        image_size=args.image_size,
                        mode='train')

        ds_valid = QaTa(dataname="Covid19_X",
                        csv_path=args.train_csv_path,
                        root_path=args.train_root_path,
                        image_size=args.image_size,
                        mode='valid')
    else:
        ds_train = QaTa(dataname="Covid19_CT",
                        csv_path=args.train_csv_path,
                        root_path=args.train_root_path,
                        image_size=args.image_size,
                        mode='train')

        ds_valid = QaTa(dataname="Covid19_CT",
                        csv_path=args.val_csv_path,
                        root_path=args.val_root_path,
                        image_size=args.image_size,
                        mode='valid')

    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)
    model = AmGANet_2DWrapper(args,task_name=args.task_name)

    session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
    model_save_path = args.task_name + '/' + args.model_name + '/' + session_name + '/'

    model_save_filename = "best_model"
    ## 1. setting recall function
    model_ckpt = ModelCheckpoint(
        dirpath=model_save_path,
        filename=model_save_filename,
        monitor='val_MIoU',
        save_top_k=1,
        mode='max',
        verbose=True,
    )

    early_stopping = EarlyStopping(monitor = 'val_MIoU',
                            patience=args.patience,
                            mode = 'max'
    )


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

