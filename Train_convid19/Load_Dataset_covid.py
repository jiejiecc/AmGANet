import json
import os
import torch
import pandas as pd
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset


class QaTa(Dataset):

    def __init__(self, dataname, csv_path=None, root_path=None, mode='train',image_size=[224,224]):

        super(QaTa, self).__init__()

        self.mode = mode
        self.dataname=dataname
        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        # self.image_list = [img.replace('mask_', '') for img in self.image_list]   # QaTa-COV19

        self.caption_list = list(self.data['Description'])
        if self.dataname =="Covid19_X":
            if mode == 'train':
                self.image_list = self.image_list[:int(0.8*len(self.image_list))]
                self.caption_list = self.caption_list[:int(0.8*len(self.caption_list))]
            elif mode == 'valid':
                self.image_list = self.image_list[int(0.8*len(self.image_list)):]
                self.caption_list = self.caption_list[int(0.8*len(self.caption_list)):]
            else:
                pass   
        elif self.dataname =="MosMedData" or self.dataname =="MoNuSeg":
            pass

        self.root_path = root_path
        self.image_size = image_size
        


    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        if self.dataname=="Covid19_X": #QaTa-COV19
            image = os.path.join(self.root_path,'images',self.image_list[idx].replace('mask_',''))#H
            gt = os.path.join(self.root_path,'labels', self.image_list[idx])

        else:  #MosMedData

            image = os.path.join(self.root_path,'images',self.image_list[idx])#H
            gt = os.path.join(self.root_path,'labels', self.image_list[idx])

        data = {'image':image, 'gt':gt}
        data = trans(data)

        image,gt = data['image'],data['gt']
        gt = torch.where(gt==255,1,0)

        return (image, gt)


    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt"]),


            ])
        
        else:  
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt"]),

            ])

        return trans
        