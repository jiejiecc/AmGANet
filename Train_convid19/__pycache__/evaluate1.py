# import argparse
# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# # 引入你的项目模块
# from engine.wrapper1 import LanGuideMedSegWrapper
# from utils.dataset1 import QaTa
# import utils.config as config

# # ==========================================
# # 1. 可视化辅助函数 (保持不变)
# # ==========================================

# def preprocess_image_for_vis(image):
#     """将 Tensor/Numpy (C, H, W) 转换为 (H, W, 3) BGR uint8 用于 OpenCV"""
#     if isinstance(image, torch.Tensor):
#         image = image.cpu().detach().numpy()
    
#     image = np.array(image).squeeze()
    
#     # 如果是 (C, H, W) 且 C=1 或 C=3
#     if image.ndim == 3:
#         if image.shape[0] == 3 or image.shape[0] == 1:
#             image = image.transpose(1, 2, 0) # (H, W, C)
    
#     # 归一化到 0-255
#     img_min, img_max = image.min(), image.max()
#     if img_max - img_min > 1e-7:
#         norm_img = (image - img_min) / (img_max - img_min)
#     else:
#         norm_img = np.zeros_like(image)
        
#     img_uint8 = (norm_img * 255).astype(np.uint8)
    
#     # 处理灰度 vs RGB
#     if img_uint8.ndim == 2:
#         img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
#     elif img_uint8.shape[2] == 1:
#          img_bgr = cv2.cvtColor(img_uint8.squeeze(), cv2.COLOR_GRAY2BGR)
#     else:
#         img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
#     return img_bgr



# # 假设 preprocess_image_for_vis 已经在外部定义

# def save_gradcam_overlay(image, cam_mask, save_path):
#     """
#     Grad-CAM 热力图叠加 (无模糊稀疏版)：
#     1. [已去除] 高斯模糊。
#     2. 指数变换 (Power)：保留稀疏聚焦效果。
#     3. 权重调整：保留较淡的背景效果。
#     """
#     # 1. 准备原图背景
#     img_bgr = preprocess_image_for_vis(image)
#     H, W = img_bgr.shape[:2]

#     # 2. 处理 CAM Mask
#     if cam_mask is None:
#         cam_mask = np.zeros((H, W), dtype=np.float32)
    
#     # 调整尺寸到原图大小
#     # 注意：没有高斯模糊，这里resize后可能会看到明显的像素格子
#     cam_mask = cv2.resize(cam_mask, (W, H))

#     # ============================================================
#     # 【已去除高斯模糊】
#     # 原代码: cam_mask = cv2.GaussianBlur(cam_mask, (13, 13), 0)
#     # ============================================================
    
#     # 归一化 (0~1)
#     c_min, c_max = cam_mask.min(), cam_mask.max()
#     if c_max - c_min > 1e-7:
#         cam_norm = (cam_mask - c_min) / (c_max - c_min)
#     else:
#         cam_norm = cam_mask

#     # 【保留】让热力图变"稀疏" (Focusing)
#     # 继续使用平方操作抑制低响应区域，让红区更集中
#     cam_norm = np.power(cam_norm, 3.0) 

#     # 3. 生成热力图
#     heatmap_uint8 = (cam_norm * 255).astype(np.uint8)
#     heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
#     # 4. 全局融合
#     # 【保留】较低的热力图权重 (0.3)，让背景蓝底比较淡，透出原图
#     overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.6, 0)
    
#     # 保存
#     save_dir = os.path.dirname(save_path)
#     if save_dir and not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)
#     cv2.imwrite(save_path, overlay)



# def save_colored_mask(image, pred, gt, save_path):
#     """
#     保存三色差异图：
#     以原图为底图，仅在预测/GT区域叠加半透明颜色。
#     颜色定义:
#     - TP (Green): 预测正确 [0, 255, 0]
#     - FN (Red):   漏检     [0, 0, 255]
#     - FP (Blue):  误检     [255, 0, 0]
#     """
#     # 1. 准备底图
#     img_bgr = preprocess_image_for_vis(image)
#     H, W = img_bgr.shape[:2]
    
#     # 2. 准备 Mask
#     pred = np.array(pred).squeeze()
#     gt = np.array(gt).squeeze()
    
#     pred = (pred > 0.5).astype(np.uint8)
#     gt = (gt > 0.5).astype(np.uint8)
    
#     # 尺寸对齐
#     if pred.shape != (H, W):
#         gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)

#     # 3. 创建纯色图层 (Color Layer)
#     # 初始化一个全黑的层
#     color_layer = np.zeros((H, W, 3), dtype=np.uint8)
    
#     # 填充颜色 (BGR 格式)
#     # TP - 绿色
#     color_layer[(pred == 1) & (gt == 1)] = [0, 255, 0] 
#     # FN - 红色 (漏检)
#     color_layer[(pred == 0) & (gt == 1)] = [0, 0, 255] 
#     # FP - 蓝色 (误检)
#     color_layer[(pred == 1) & (gt == 0)] = [255, 0, 0] 

#     # 4. 局部融合逻辑
#     # 找出有颜色的区域
#     mask_indices = np.any(color_layer != [0, 0, 0], axis=-1)
    
#     # 复制原图
#     vis_img = img_bgr.copy()
    
#     # 在有颜色的区域：原图 60% + 颜色 40%
#     # 在无颜色的区域：保持原图 100%
#     vis_img[mask_indices] = cv2.addWeighted(
#         img_bgr[mask_indices], 0.5, 
#         color_layer[mask_indices], 1, 
#         0
#     ).squeeze()

#     # 5. 保存
#     cv2.imwrite(save_path, vis_img)

# # ==========================================
# # 2. Grad-CAM 类 (针对你的输入逻辑修改)
# # ==========================================

# class SemanticSegmentationGradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         # 注册 hook
#         self.handle_f = self.target_layer.register_forward_hook(self.save_activation)
#         self.handle_b = self.target_layer.register_full_backward_hook(self.save_gradient)

#     def save_activation(self, module, input, output):
#         self.activations = output.detach()

#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()

#     def remove_hooks(self):
#         self.handle_f.remove()
#         self.handle_b.remove()

#     def __call__(self, input_tensor):
#         self.model.eval()
#         self.model.zero_grad()

#         # 1. Forward Pass
#         # 直接传入 Tensor，因为你的 AgileFormer2D.forward 接受 x (image tensor)
#         output = self.model(input_tensor) 
        
#         # 处理输出可能是 list 或 dict 的情况
#         if isinstance(output, dict):
#             output = output['pred_mask']
#         elif isinstance(output, (list, tuple)):
#             output = output[0] # 假设第一个是预测结果
        
#         pred_prob = torch.sigmoid(output)
#         mask = (pred_prob > 0.5).float()
        
#         if mask.sum() == 0:
#             return None

#         # 2. Backward Pass
#         loss = (output * mask).sum()
#         loss.backward()

#         if self.gradients is None or self.activations is None:
#             return None

#         # 3. Generate CAM
#         weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
#         cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
#         cam = F.relu(cam)
        
#         cam = cam.cpu().numpy()
#         cam = cam[0, 0, :, :] 
        
#         # Normalize
#         cam_min, cam_max = np.min(cam), np.max(cam)
#         if cam_max - cam_min > 1e-7:
#             cam = (cam - cam_min) / (cam_max - cam_min)
#         else:
#             cam = np.zeros_like(cam)
            
#         return cam.astype(np.float32)


# # ==========================================
# # 3. 主程序
# # ==========================================

# def get_parser():
#     parser = argparse.ArgumentParser(description='Language-guide Medical Image Segmentation')
#     # 请确保这里的 config 路径是正确的
#     parser.add_argument('--config', default='LanGuideMedSeg-MICCAI2023-main/config/training.yaml', type=str, help='config file')
#     args = parser.parse_args()
#     assert args.config is not None
#     cfg = config.load_cfg_from_cfg_file(args.config)
#     return cfg

# if __name__ == '__main__':
#     args = get_parser()

#     # 1. 加载模型
#     model = LanGuideMedSegWrapper(args)
#     ckpt_path = 'LanGuideMedSeg-MICCAI2023-main/QaTasave_model1/attributive_biomedclipmedseg_QaTa-COV19.ckpt'
#     print(f"Loading checkpoint: {ckpt_path}")
#     checkpoint = torch.load(ckpt_path, map_location='cpu')["state_dict"]
#     model.load_state_dict(checkpoint, strict=True)
#     model = model.cuda()
#     model.eval()

#     # 2. 数据集设置 (确保你的 utils/dataset1.py 已经保存了修改)
#     ds_test = QaTa(dataname="QaTa",
#                     csv_path=args.test_csv_path,
#                     root_path=args.test_root_path,
#                     image_size=args.image_size,
#                     mode='test')
#     # ds_test = QaTa(dataname="MosMedData",
#     #                 csv_path=args.test_csv_path,
#     #                 root_path=args.test_root_path,
#     #                 image_size=args.image_size,
#     #                 mode='test')
#     # 注意：batch_size=1 时，文件名会是一个长度为1的元组
#     dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4)

#     # 3. 创建保存文件夹
#     vis_root = "visualize_finalX"
#     path_gt = os.path.join(vis_root, 'GT')
#     path_cam = os.path.join(vis_root, 'GradCAM')
#     path_pred = os.path.join(vis_root, 'Prediction')
#     for p in [path_gt, path_cam, path_pred]:
#         os.makedirs(p, exist_ok=True)

#     # 4. Grad-CAM 目标层
#     # try:
#     #     # target_layer = model.model.agile_former.norm_up
#     #     # target_layer = model.model.agile_former.output
#     #     target_layer = model.model.agile_former.stages_up[-1]

#     # except AttributeError:
#     #     target_layer = model.model.agile_former.up_projs[-1]

#     target_layer = model.model.agile_former.up_projs[-1]

#     # 5. 推理循环
#     print(f"Start processing {len(ds_test)} images...")
#     grad_cam = SemanticSegmentationGradCAM(model, target_layer)

#     dice_scores = []

#     for i, batch in enumerate(tqdm(dl_test)):
#         # ========================================================
#         # 【关键修改】：正确解包三个返回值
#         # Dataset 返回: (image, gt, image_name)
#         # DataLoader batch 结构: [image_batch, gt_batch, name_tuple]
#         # ========================================================
        
#         img_tensor = batch[0].cuda()
#         gt_mask = batch[1] # 掩码保持在 CPU 转 numpy
        
#         # 获取文件名
#         if len(batch) >= 3:
#             raw_name = batch[2][0] # 取出元组中的第一个字符串
#             # 去掉后缀 (例如 "case_01.png" -> "case_01")
#             img_name = os.path.splitext(raw_name)[0]
#         else:
#             # 如果 Dataset 没改对，回退到默认命名
#             img_name = f"img_{i}"

#         # --------------------------------------------------------
#         # 后续逻辑保持不变
#         # --------------------------------------------------------
        
#         # 预处理
#         img_vis_bgr = preprocess_image_for_vis(img_tensor)
#         gt_mask_np = gt_mask.cpu().detach().numpy()

#         # A. 保存 GT
#         gt_vis = (gt_mask_np.squeeze() * 255).astype(np.uint8)
#         cv2.imwrite(os.path.join(path_gt, f"{img_name}_gt.png"), gt_vis)

#         # B. 保存 Grad-CAM
#         with torch.set_grad_enabled(True):
#             img_tensor.requires_grad = True
#             cam_result = grad_cam(img_tensor) 
#             img_tensor.requires_grad = False
        
#         save_gradcam_overlay(img_vis_bgr, cam_result, 
#                              os.path.join(path_cam, f"{img_name}_cam.jpg"))
        
#         # C. 保存 预测误差图
#         with torch.no_grad():
#             output = model(img_tensor)
#             if isinstance(output, dict):
#                 pred_logits = output.get('pred_mask', list(output.values())[0])
#             elif isinstance(output, (list, tuple)):
#                 pred_logits = output[0]
#             else:
#                 pred_logits = output

#             if not isinstance(pred_logits, torch.Tensor): continue

#             pred_prob = torch.sigmoid(pred_logits)
#             pred_mask = (pred_prob > 0.5).float().cpu().numpy()

#         save_colored_mask(img_vis_bgr, pred_mask, gt_mask_np, 
#                           os.path.join(path_pred, f"{img_name}_pred.png"))

#         # D. Dice 计算
#         intersection = np.sum(gt_mask_np * pred_mask)
#         union = np.sum(gt_mask_np) + np.sum(pred_mask)
#         dice = 2 * intersection / (union + 1e-5)
#         dice_scores.append(dice)

#     grad_cam.remove_hooks()
#     print(f"Visualization Done. Average Dice: {np.mean(dice_scores):.4f}")



# import argparse
# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# # 引入你的项目模块
# from engine.wrapper1 import LanGuideMedSegWrapper
# from utils.dataset1 import QaTa
# import utils.config as config

# # ==========================================
# # 1. 可视化辅助函数
# # ==========================================

# def preprocess_image_for_vis(image):
#     if isinstance(image, torch.Tensor):
#         image = image.cpu().detach().numpy()
    
#     image = np.array(image).squeeze()
    
#     if image.ndim == 3:
#         if image.shape[0] == 3 or image.shape[0] == 1:
#             image = image.transpose(1, 2, 0)
    
#     img_min, img_max = image.min(), image.max()
#     if img_max - img_min > 1e-7:
#         norm_img = (image - img_min) / (img_max - img_min)
#     else:
#         norm_img = np.zeros_like(image)
        
#     img_uint8 = (norm_img * 255).astype(np.uint8)
    
#     if img_uint8.ndim == 2:
#         img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
#     elif img_uint8.shape[2] == 1:
#         img_bgr = cv2.cvtColor(img_uint8.squeeze(), cv2.COLOR_GRAY2BGR)
#     else:
#         img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
#     return img_bgr

# def save_gradcam_overlay(image, cam_mask, save_path):
#     """
#     Grad-CAM 可视化：加入了适当的归一化防止全红
#     """
#     img_bgr = preprocess_image_for_vis(image)
#     H, W = img_bgr.shape[:2]

#     if cam_mask is None:
#         cam_mask = np.zeros((H, W), dtype=np.float32)
    
#     # 插值
#     cam_mask = cv2.resize(cam_mask, (W, H))
    
#     # 简单的平滑，去掉 Transformer 的格子
#     cam_mask = cv2.GaussianBlur(cam_mask, (7, 7), 0)
    
#     # 归一化
#     c_min, c_max = cam_mask.min(), cam_mask.max()
#     if c_max - c_min > 1e-7:
#         cam_norm = (cam_mask - c_min) / (c_max - c_min)
#     else:
#         cam_norm = cam_mask

#     cam_norm = np.power(cam_norm, 3.0) 
#     heatmap_uint8 = (cam_norm * 255).astype(np.uint8)
#     heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
#     # 融合
#     overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.6, 0)
    
#     save_dir = os.path.dirname(save_path)
#     if save_dir and not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)
#     cv2.imwrite(save_path, overlay)

# def save_colored_mask(image, pred, gt, save_path):
#     img_bgr = preprocess_image_for_vis(image)
#     H, W = img_bgr.shape[:2]
    
#     pred = np.array(pred).squeeze()
#     gt = np.array(gt).squeeze()
#     pred = (pred > 0.5).astype(np.uint8)
#     gt = (gt > 0.5).astype(np.uint8)
    
#     if pred.shape != (H, W):
#         gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)

#     color_layer = np.zeros((H, W, 3), dtype=np.uint8)
#     color_layer[(pred == 1) & (gt == 1)] = [0, 255, 0] # TP
#     color_layer[(pred == 0) & (gt == 1)] = [0, 0, 255] # FN
#     color_layer[(pred == 1) & (gt == 0)] = [255, 0, 0] # FP

#     mask_indices = np.any(color_layer != [0, 0, 0], axis=-1)
#     vis_img = img_bgr.copy()
#     vis_img[mask_indices] = cv2.addWeighted(
#         img_bgr[mask_indices], 0.5, 
#         color_layer[mask_indices], 1, 
#         0
#     ).squeeze()

#     cv2.imwrite(save_path, vis_img)

# # ==========================================
# # 2. Grad-CAM 类 (关键修改：兼容 tuple 返回值)
# # ==========================================

# class SemanticSegmentationGradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         # 注册 hook
#         self.handle_f = self.target_layer.register_forward_hook(self.save_activation)
#         self.handle_b = self.target_layer.register_full_backward_hook(self.save_gradient)

#     def save_activation(self, module, input, output):
#         # 【修改】如果输出是元组 (x, text)，只取 x
#         if isinstance(output, (list, tuple)):
#             output = output[0]
#         self.activations = output.detach()

#     def save_gradient(self, module, grad_input, grad_output):
#         # 【修改】梯度通常也是元组，取第一个
#         g = grad_output[0]
#         if isinstance(g, (list, tuple)):
#             g = g[0]
#         self.gradients = g.detach()

#     def remove_hooks(self):
#         self.handle_f.remove()
#         self.handle_b.remove()

#     def __call__(self, input_tensor):
#         self.model.eval()
#         self.model.zero_grad()

#         # 1. Forward Pass
#         output = self.model(input_tensor) 
        
#         if isinstance(output, dict):
#             output = output['pred_mask']
#         elif isinstance(output, (list, tuple)):
#             output = output[0]
        
#         pred_prob = torch.sigmoid(output)
#         mask = (pred_prob > 0.5).float()
        
#         # 如果没有预测结果，就不画热力图
#         if mask.sum() == 0:
#             return None

#         # 2. Backward Pass
#         loss = (output * mask).sum()
#         loss.backward()

#         if self.gradients is None or self.activations is None:
#             return None

#         # 3. Generate CAM
#         # 这里使用标准的 Grad-CAM (Global Average Pooling)
#         weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
#         cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
#         cam = F.relu(cam)
        
#         cam = cam.cpu().numpy()
#         cam = cam[0, 0, :, :] 
        
#         # Normalize
#         cam_min, cam_max = np.min(cam), np.max(cam)
#         if cam_max - cam_min > 1e-7:
#             cam = (cam - cam_min) / (cam_max - cam_min)
#         else:
#             cam = np.zeros_like(cam)
            
#         return cam.astype(np.float32)

# # ==========================================
# # 3. 主程序
# # ==========================================

# def get_parser():
#     parser = argparse.ArgumentParser(description='Language-guide Medical Image Segmentation')
#     parser.add_argument('--config', default='LanGuideMedSeg-MICCAI2023-main/config/training.yaml', type=str, help='config file')
#     args = parser.parse_args()
#     assert args.config is not None
#     cfg = config.load_cfg_from_cfg_file(args.config)
#     return cfg

# if __name__ == '__main__':
#     args = get_parser()

#     # 1. 加载模型
#     model = LanGuideMedSegWrapper(args)
#     ckpt_path = 'LanGuideMedSeg-MICCAI2023-main/QaTasave_model1/attributive_biomedclipmedseg_QaTa-COV19.ckpt'
#     print(f"Loading checkpoint: {ckpt_path}")
#     checkpoint = torch.load(ckpt_path, map_location='cpu')["state_dict"]
#     model.load_state_dict(checkpoint, strict=True)
#     model = model.cuda()
#     model.eval()

#     # 2. 数据集设置
#     ds_test = QaTa(dataname="QaTa", # 确保 dataname 对应您的代码逻辑
#                    csv_path=args.test_csv_path,
#                    root_path=args.test_root_path,
#                    image_size=args.image_size,
#                    mode='test')
#     # ds_test = QaTa(dataname="MosMedData",
#     #                 csv_path=args.test_csv_path,
#     #                 root_path=args.test_root_path,
#     #                 image_size=args.image_size,
#     #                 mode='test')
#     dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4)

#     # 3. 创建保存文件夹
#     vis_root = "visualize_X" # 改名区分
#     path_gt = os.path.join(vis_root, 'GT')
#     path_cam = os.path.join(vis_root, 'GradCAM')
#     path_pred = os.path.join(vis_root, 'Prediction')
#     for p in [path_gt, path_cam, path_pred]:
#         os.makedirs(p, exist_ok=True)

#     # 4. 设置 Grad-CAM 目标层为 stages_up[-1]
#     # 注意：stages_up 包含多个 Decoder Block
#     try:
#         target_layer = model.model.agile_former.stages_up[-1]
#         print(f"Successfully hooked: stages_up[-1]")
#     except AttributeError:
#         print("Error: Could not find stages_up. Check model structure.")
#         exit()

#     # 5. 推理循环
#     print(f"Start processing {len(ds_test)} images...")
#     grad_cam = SemanticSegmentationGradCAM(model, target_layer)

#     dice_scores = []

#     for i, batch in enumerate(tqdm(dl_test)):
        
#         img_tensor = batch[0].cuda()
#         gt_mask = batch[1]
        
#         if len(batch) >= 3:
#             raw_name = batch[2][0]
#             # 兼容 mask_ 前缀
#             clean_name = raw_name.replace('mask_', '')
#             img_name = os.path.splitext(clean_name)[0]
#         else:
#             img_name = f"img_{i}"

#         img_vis_bgr = preprocess_image_for_vis(img_tensor)
#         gt_mask_np = gt_mask.cpu().detach().numpy()

#         # A. 保存 GT
#         gt_vis = (gt_mask_np.squeeze() * 255).astype(np.uint8)
#         cv2.imwrite(os.path.join(path_gt, f"{img_name}_gt.png"), gt_vis)

#         # B. 保存 Grad-CAM (使用 stages_up[-1])
#         with torch.set_grad_enabled(True):
#             img_tensor.requires_grad = True
#             cam_result = grad_cam(img_tensor) 
#             img_tensor.requires_grad = False
        
#         save_gradcam_overlay(img_vis_bgr, cam_result, 
#                              os.path.join(path_cam, f"{img_name}_cam.jpg"))
        
#         # C. 保存 预测误差图
#         with torch.no_grad():
#             output = model(img_tensor)
#             if isinstance(output, dict):
#                 pred_logits = output.get('pred_mask', list(output.values())[0])
#             elif isinstance(output, (list, tuple)):
#                 pred_logits = output[0]
#             else:
#                 pred_logits = output

#             if not isinstance(pred_logits, torch.Tensor): continue

#             pred_prob = torch.sigmoid(pred_logits)
#             pred_mask = (pred_prob > 0.5).float().cpu().numpy()

#         save_colored_mask(img_vis_bgr, pred_mask, gt_mask_np, 
#                           os.path.join(path_pred, f"{img_name}_pred.png"))

#         # D. Dice 计算
#         intersection = np.sum(gt_mask_np * pred_mask)
#         union = np.sum(gt_mask_np) + np.sum(pred_mask)
#         dice = 2 * intersection / (union + 1e-5)
#         dice_scores.append(dice)

#     grad_cam.remove_hooks()
#     print(f"Visualization Done. Average Dice: {np.mean(dice_scores):.4f}")


import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

# 引入你的项目模块
from engine.wrapper1 import LanGuideMedSegWrapper
from utils.dataset1 import QaTa
import utils.config as config

# ==========================================
# 1. 可视化辅助函数
# ==========================================

def preprocess_image_for_vis(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
    
    image = np.array(image).squeeze()
    
    if image.ndim == 3:
        if image.shape[0] == 3 or image.shape[0] == 1:
            image = image.transpose(1, 2, 0)
    
    img_min, img_max = image.min(), image.max()
    if img_max - img_min > 1e-7:
        norm_img = (image - img_min) / (img_max - img_min)
    else:
        norm_img = np.zeros_like(image)
        
    img_uint8 = (norm_img * 255).astype(np.uint8)
    
    if img_uint8.ndim == 2:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    elif img_uint8.shape[2] == 1:
        img_bgr = cv2.cvtColor(img_uint8.squeeze(), cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
    return img_bgr

def save_gradcam_overlay(image, cam_mask, save_path):
    """
    2D Grad-CAM 叠加图
    处理逻辑：Resize -> Normalize -> Power(3.0) -> Colormap
    """
    img_bgr = preprocess_image_for_vis(image)
    H, W = img_bgr.shape[:2]

    if cam_mask is None:
        cam_mask = np.zeros((H, W), dtype=np.float32)
    
    # 1. 插值
    cam_mask = cv2.resize(cam_mask, (W, H))
    
    # 2. 简单的平滑 (2D图通常需要一点模糊才好看)
    cam_mask = cv2.GaussianBlur(cam_mask, (7, 7), 0)
    
    # 3. 归一化
    c_min, c_max = cam_mask.min(), cam_mask.max()
    if c_max - c_min > 1e-7:
        cam_norm = (cam_mask - c_min) / (c_max - c_min)
    else:
        cam_norm = cam_mask

    # 4. 【关键】3次方增强，聚焦热点
    # cam_norm = np.power(cam_norm, 3.0) 
    
    heatmap_uint8 = (cam_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 5. 融合
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.6, 0)
    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_path, overlay)

from matplotlib.ticker import MultipleLocator
def save_3d_surface_plot_final(heatmap_2d, save_path):
    """
    3D 表面图 (修复扁平问题和颜色条遮挡问题)
    """
    # ==========================
    # 1. 数据预处理 (保持不变)
    # ==========================
    # 尺寸对齐
    target_size = (224, 224)
    if heatmap_2d.shape != target_size:
        heatmap_2d = cv2.resize(heatmap_2d, target_size, interpolation=cv2.INTER_NEAREST)

    # 归一化
    h_min, h_max = heatmap_2d.min(), heatmap_2d.max()
    if h_max - h_min > 1e-7:
        heatmap_norm = (heatmap_2d - h_min) / (h_max - h_min)
    else:
        heatmap_norm = heatmap_2d

    # 3次方增强
    heatmap_processed = np.power(heatmap_norm, 3.0)
    
    # 背景净化
    threshold = 0.05
    heatmap_processed[heatmap_processed < threshold] = 0

    # 准备绘图数据
    H, W = heatmap_processed.shape
    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    X, Y = np.meshgrid(x, y)
    Z = heatmap_processed

    # ==========================
    # 2. 创建画布与绘制
    # ==========================
    fig = plt.figure(figsize=(12, 9), dpi=120) # 稍微增加画布高度
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 (rstride=1 保留像素感)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False,
                           rstride=1, cstride=1)

    # ==========================
    # 3. 视角与外观设置 (关键修改区域)
    # ==========================
    # 视角 (保持之前的设置)
    ax.view_init(elev=30, azim=-45)
    
    # 网格设置
    ax.grid(True)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.zaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_zlim(0, 2.5)

    # 【修复 1：解决扁平问题】
    # 调整盒子的长宽高比例 (X, Y, Z)
    # 之前的 0.4 改为 0.75，让 Z 轴看起来接近 X/Y 轴高度的 3/4
    # 如果觉得还不够高，可以改成 (1, 1, 1)
    ax.set_box_aspect((1, 1, 0.75)) 

    # 【修复 2：解决颜色条挡住坐标问题】
    # pad: 颜色条与主图轴之间的距离。增加 pad (例如从 0.02 改到 0.12) 可以把它往右推。
    # shrink: 稍微调大一点 (0.6 -> 0.7)，因为图变高了，颜色条也跟着高一点才协调。
    fig.colorbar(surf, ax=ax, shrink=0.8, aspect=15, fraction=0.03, pad=0.05)

    # ==========================
    # 4. 保存
    # ==========================
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 增加 pad_inches 防止边缘被裁切
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()

def save_colored_mask(image, pred, gt, save_path):
    img_bgr = preprocess_image_for_vis(image)
    H, W = img_bgr.shape[:2]
    
    pred = np.array(pred).squeeze()
    gt = np.array(gt).squeeze()
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    
    if pred.shape != (H, W):
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)

    color_layer = np.zeros((H, W, 3), dtype=np.uint8)
    color_layer[(pred == 1) & (gt == 1)] = [0, 255, 0] # TP
    color_layer[(pred == 0) & (gt == 1)] = [0, 0, 255] # FN
    color_layer[(pred == 1) & (gt == 0)] = [255, 0, 0] # FP

    mask_indices = np.any(color_layer != [0, 0, 0], axis=-1)
    vis_img = img_bgr.copy()
    vis_img[mask_indices] = cv2.addWeighted(
        img_bgr[mask_indices], 0.5, 
        color_layer[mask_indices], 1, 
        0
    ).squeeze()

    cv2.imwrite(save_path, vis_img)

# ==========================================
# 2. Grad-CAM 类
# ==========================================

class SemanticSegmentationGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.handle_f = self.target_layer.register_forward_hook(self.save_activation)
        self.handle_b = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        if isinstance(output, (list, tuple)):
            output = output[0]
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        g = grad_output[0]
        if isinstance(g, (list, tuple)):
            g = g[0]
        self.gradients = g.detach()

    def remove_hooks(self):
        self.handle_f.remove()
        self.handle_b.remove()

    def __call__(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor) 
        
        if isinstance(output, dict):
            output = output['pred_mask']
        elif isinstance(output, (list, tuple)):
            output = output[0]
        
        pred_prob = torch.sigmoid(output)
        mask = (pred_prob > 0.5).float()
        
        if mask.sum() == 0:
            return None

        loss = (output * mask).sum()
        loss.backward()

        if self.gradients is None or self.activations is None:
            return None

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.cpu().numpy()
        cam = cam[0, 0, :, :] 
        
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam.astype(np.float32)

# ==========================================
# 3. 主程序
# ==========================================

def get_parser():
    parser = argparse.ArgumentParser(description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config', default='LanGuideMedSeg-MICCAI2023-main/config/training.yaml', type=str, help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

if __name__ == '__main__':
    args = get_parser()

    # 1. 加载模型
    model = LanGuideMedSegWrapper(args)
    ckpt_path = 'QaTasave_modeljiedong/attributive_biomedclipmedseg_QaTa.ckpt'
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model = model.cuda()
    model.eval()

    # 2. 数据集设置
    ds_test = QaTa(dataname="QaTa", 
                   csv_path=args.test_csv_path,
                   root_path=args.test_root_path,
                   image_size=args.image_size,
                   mode='test')
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4)

    # 3. 创建保存文件夹
    vis_root = "visualize_xiaorong1" 
    path_gt = os.path.join(vis_root, 'GT')
    path_cam = os.path.join(vis_root, 'GradCAM')
    path_pred = os.path.join(vis_root, 'Prediction')
    path_3d = os.path.join(vis_root, '3D_Surface')
    
    for p in [path_gt, path_cam, path_pred, path_3d]:
        os.makedirs(p, exist_ok=True)

    # 4. Grad-CAM 目标层
    try:
        target_layer = model.model.agile_former.stages_up[-1]
        # target_layer = model.model.agile_former.up_projs[-2]

        print(f"Successfully hooked: stages_up[-1]")
    except AttributeError:
        print("Error: Could not find stages_up. Check model structure.")
        exit()

    # 5. 推理循环
    print(f"Start processing {len(ds_test)} images...")
    grad_cam = SemanticSegmentationGradCAM(model, target_layer)

    dice_scores = []

    for i, batch in enumerate(tqdm(dl_test)):
        
        img_tensor = batch[0].cuda()
        gt_mask = batch[1]
        
        if len(batch) >= 3:
            raw_name = batch[2][0]
            clean_name = raw_name.replace('mask_', '')
            img_name = os.path.splitext(clean_name)[0]
        else:
            img_name = f"img_{i}"

        img_vis_bgr = preprocess_image_for_vis(img_tensor)
        gt_mask_np = gt_mask.cpu().detach().numpy()

        # A. 保存 GT
        gt_vis = (gt_mask_np.squeeze() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(path_gt, f"{img_name}_gt.png"), gt_vis)

        # ----------------------------------------------------------------
        # B. 获取 Grad-CAM 结果 (核心步骤)
        # ----------------------------------------------------------------
        with torch.set_grad_enabled(True):
            img_tensor.requires_grad = True
            # 这里计算出来的 cam_result 是原始的 Grad-CAM 归一化结果 (0-1)
            cam_result = grad_cam(img_tensor) 
            img_tensor.requires_grad = False
        
        # 1. 传给 2D 绘图 (内部会做 resize -> smooth -> power(3) -> draw)
        save_gradcam_overlay(img_vis_bgr, cam_result, 
                             os.path.join(path_cam, f"{img_name}_cam.jpg"))

        # 2. 传给 3D 绘图 (内部会做 resize -> power(3) -> draw)
        # 如果 cam_result 是 None (说明没预测到)，则不画
        if cam_result is not None:
            save_3d_surface_plot_final(cam_result, 
                                       os.path.join(path_3d, f"{img_name}_3d.png"))

        # ----------------------------------------------------------------
        
        # C. 保存 预测误差图
        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, dict):
                pred_logits = output.get('pred_mask', list(output.values())[0])
            elif isinstance(output, (list, tuple)):
                pred_logits = output[0]
            else:
                pred_logits = output

            if not isinstance(pred_logits, torch.Tensor): continue

            pred_prob = torch.sigmoid(pred_logits)
            pred_mask = (pred_prob > 0.5).float().cpu().numpy()

        save_colored_mask(img_vis_bgr, pred_mask, gt_mask_np, 
                          os.path.join(path_pred, f"{img_name}_pred.png"))

        # D. Dice 计算
        intersection = np.sum(gt_mask_np * pred_mask)
        union = np.sum(gt_mask_np) + np.sum(pred_mask)
        dice = 2 * intersection / (union + 1e-5)
        dice_scores.append(dice)

    grad_cam.remove_hooks()
    print(f"Visualization Done. Average Dice: {np.mean(dice_scores):.4f}")