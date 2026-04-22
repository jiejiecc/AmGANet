import os
import sys
import warnings
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Load_Dataset_BUSI import ValGenerator, ImageToImage2D
import Config_BUSI as config
from utils import *

warnings.filterwarnings("ignore")

# 固定输入尺寸时通常会更快
torch.backends.cudnn.benchmark = True


def compute_dice_iou(predict_save, labs):
    if isinstance(predict_save, torch.Tensor):
        predict_save = predict_save.detach().cpu().numpy()
    if isinstance(labs, torch.Tensor):
        labs = labs.detach().cpu().numpy()

    predict_save = np.squeeze(predict_save)
    labs = np.squeeze(labs)

    tmp_pred = (predict_save > 0).astype(np.float32)
    tmp_lbl = (labs > 0).astype(np.float32)

    if tmp_pred.shape != tmp_lbl.shape:
        raise ValueError(f"Shape mismatch: pred {tmp_pred.shape} vs label {tmp_lbl.shape}")

    lbl_sum = np.sum(tmp_lbl)
    pred_sum = np.sum(tmp_pred)
    inter = np.sum(tmp_lbl * tmp_pred)
    union = lbl_sum + pred_sum - inter

    if lbl_sum == 0 and pred_sum == 0:
        dice_pred = 1.0
        iou_pred = 1.0
    else:
        dice_pred = 2.0 * inter / (lbl_sum + pred_sum + 1e-5)
        iou_pred = inter / (union + 1e-5)

    return dice_pred, iou_pred, tmp_pred


def save_mask(mask, save_path):
    save_img = (mask * 255).astype(np.uint8)

    if config.task_name == "MoNuSeg":
        save_img = cv2.pyrUp(save_img, dstsize=(448, 448))
        save_img = cv2.resize(save_img, (2000, 2000), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(save_path, save_img)


def infer_one(model, input_img):
    # input_img: already on cuda
    output = model(input_img)

    # 只有输出不是概率时才做 sigmoid
    if output.min().item() < 0.0 or output.max().item() > 1.0:
        output = torch.sigmoid(output)

    pred = (output > 0.5).float()
    return pred


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    test_session = config.test_session
    model_type = config.model_name

    # 是否保存可视化；纯测试建议 False，会明显快很多

    SAVE_VIS = True

    if config.task_name == "BUSI":
        test_num = 78
        model_path = f"./BUSI/{model_type}/{test_session}/models/best_model-{model_type}.pth.tar"
    elif config.task_name == "Kvasir":
        test_num = 100
        model_path = f"./Kvasir/{model_type}/{test_session}/models/best_model-{model_type}.pth.tar"
    else:
        raise ValueError(f"Unsupported task_name: {config.task_name}")

    vis_path = f"./{config.task_name}_visualize_test/"
    os.makedirs(vis_path, exist_ok=True)

    checkpoint = torch.load(model_path, map_location='cuda')


    if model_type == 'AmGANet':
        from model.AmGANet_2D import AmGANet_2D
        import argparse
        from model.config import get_config

        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str,
                            metavar="FILE",
                            default="./model/configs/AmGANet.yaml",
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
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    print('Model loaded!')

    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(
        config.test_dataset,
        config.task_name,
        tf_test,
        image_size=config.img_size
    )

    # 这里是关键提速点
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,      # 可试 2 / 4 / 8
        pin_memory=True,
        drop_last=False
    )

    dice_pred = 0.0
    iou_pred = 0.0

    with torch.inference_mode():
        with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
            for i, (sampled_batch, names) in enumerate(test_loader, 1):
                # 不再 tensor -> numpy -> tensor 来回转
                input_img = sampled_batch['image'].cuda(non_blocking=True)
                lab = sampled_batch['label'].cpu().numpy()

                pred_class = infer_one(model, input_img)
                predict_save = pred_class[0].detach().cpu().numpy().squeeze()

                dice_pred_t, iou_pred_t, pred_mask = compute_dice_iou(predict_save, lab)

                if SAVE_VIS:
                    # 直接用 cv2 保存，比 matplotlib.savefig 快很多
                    label_mask = (np.squeeze(lab) > 0).astype(np.uint8)
                    save_mask(label_mask, os.path.join(vis_path, f"{names}_lab.jpg"))
                    save_mask(pred_mask, os.path.join(vis_path, f"{names}_predict{model_type}.jpg"))

                dice_pred += dice_pred_t
                iou_pred += iou_pred_t
                pbar.update()

    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)