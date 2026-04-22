import argparse
from engine.wrapper1 import LanGuideMedSegWrapper

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl  

from utils.dataset1 import QaTa
import utils.config as config

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='LanGuideMedSeg-MICCAI2023-main/config/training.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg

if __name__ == '__main__':

    args = get_parser()

    # load model
    model = LanGuideMedSegWrapper(args)
# LanGuideMedSeg-MICCAI2023-main/save_model/attributive_biomedclipmedseg_QaTa-COV19.ckpt
    checkpoint = torch.load('LanGuideMedSeg-MICCAI2023-main/QaTaDatasave_model/attributive_biomedclipmedseg_QaTa.ckpt',map_location='cpu')["state_dict"]
    # model.load_state_dict(checkpoint,strict=True)
    model.load_state_dict(checkpoint,strict=False)

    # dataloader
    ds_test = QaTa(dataname="QaTa",
                    csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    image_size=args.image_size,
                    mode='test')
    
    # ds_test = QaTa(dataname="MosMedData",
    #                 csv_path=args.test_csv_path,
    #                 root_path=args.test_root_path,
    #                 image_size=args.image_size,
    #                 mode='test')
    # ds_test = QaTa(dataname="MoNuSeg",
    #                 csv_path=args.test_csv_path,
    #                 root_path=args.test_root_path,
    #                 image_size=args.image_size,
    #                 mode='test')
    # ds_test = QaTa(dataname="BUSI",
    #                 root_path=args.test_root_path,
    #                 image_size=args.image_size,
    #                 mode='test')
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)
    # dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=8)

# args.valid_batch_size
    trainer = pl.Trainer(accelerator='gpu',devices=1) 
    model.eval()
    trainer.test(model, dl_test) 




