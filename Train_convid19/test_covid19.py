
import argparse
from wrapper import AmGANet_2DWrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl  

from Load_Dataset_covid import QaTa
import config_Covid19 as config

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

if __name__ == '__main__':

    args = get_parser()

    # load model
    model = AmGANet_2DWrapper(args, task_name=args.task_name)

    checkpoint = torch.load('Covid19_CT/AmGANet/Test_session_04.22_12h59/best_model.ckpt',map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint,strict=True)

    # dataloader
    if args.task_name == "Covid19_X":

        ds_test = QaTa(dataname="Covid19_X",
                        csv_path=args.test_csv_path,
                        root_path=args.test_root_path,
                        image_size=args.image_size,
                        mode='test')
    else:
        ds_test = QaTa(dataname="Covid19_CT",
                        csv_path=args.test_csv_path,
                        root_path=args.test_root_path,
                        image_size=args.image_size,
                        mode='test')
    
   
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)

# args.valid_batch_size
    trainer = pl.Trainer(accelerator='gpu',devices=1) 
    model.eval()
    trainer.test(model, dl_test) 




