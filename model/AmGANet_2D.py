import argparse
import torch
import torch.nn as nn

from .AmGANet import AmGANet

def load_pretrained(ckpt_path, model):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    msg = model.load_pretrained(checkpoint['model'])
    # print(msg)
    del checkpoint
    torch.cuda.empty_cache()


class AmGANet_2D(nn.Module):
    def __init__(self, config, task_name, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.agile_former = AmGANet(**config.MODEL.Params, task_name=task_name, num_classes=num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.agile_former(x)
        if self.num_classes == 1:
            return torch.sigmoid_(logits)

        else:
            return logits


    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        load_pretrained(pretrained_path, self.agile_former)


