# import torch
# import torch.nn as nn

# try:
#     # raise Exception("not work")
#     from tvdcn.ops import PackedDeformConv2d, PackedDeformConv3d
#     print("tvdcn is installed, using it for deformable convolution")
    
#     class DeformConv2d(PackedDeformConv2d):
#         def __init__(self, in_channels, out_channels, kernel_size, 
#                     stride=1, padding=0, dilation=1, groups=1, 
#                     offset_groups=1, mask_groups=1, bias=True, 
#                     generator_bias: bool = False, 
#                     deformable: bool = True, modulated: bool = False):
#             super().__init__(in_channels, out_channels, kernel_size, 
#                             stride, padding, dilation, groups, offset_groups, 
#                             mask_groups, bias, generator_bias, deformable, modulated)

#         def forward(self, x):
#             return super().forward(x)


#     class DeformConv3d(PackedDeformConv3d):
#         def __init__(self, in_channels, out_channels, kernel_size, 
#                     stride=1, padding=0, dilation=1, groups=1, 
#                     offset_groups=1, mask_groups=1, bias=True, 
#                     generator_bias: bool = False, 
#                     deformable: bool = True, modulated: bool = False):
#             super().__init__(in_channels, out_channels, kernel_size, 
#                             stride, padding, dilation, groups, offset_groups, 
#                             mask_groups, bias, generator_bias, deformable, modulated)

#         def forward(self, x):
#             return super().forward(x)
# except:
#     try:
#         # raise Exception("not work")
#         from mmcv.ops import DeformConv2dPack
#         print("tvdcn is not installed, using mmcv for deformable convolution")
#         class DeformConv2d(DeformConv2dPack):
#             def __init__(self, *args, **kwargs):
#                 super().__init__(*args, **kwargs)

#             def forward(self, x):
#                 x = x.contiguous()
#                 return super().forward(x)
#     except:
#         from torchvision.ops import deform_conv2d
#         print("Neither tvdcn nor mmcv is not installed, using torchvision for deformable convolution")
#         class DeformConv2d(nn.Conv2d):
#             def __init__(self, *args, **kwargs):
#                 super().__init__(*args, **kwargs)
#                 self.offset_conv = nn.Conv2d(self.in_channels,
#                                          2 * self.kernel_size[0] * self.kernel_size[0],
#                                          kernel_size=self.kernel_size,
#                                          stride=self.stride,
#                                          padding=self.padding)

#                 nn.init.constant_(self.offset_conv.weight, 0.)
#                 nn.init.constant_(self.offset_conv.bias, 0.)

#             def forward(self, x):
#                 offset = self.offset_conv(x)
#                 x = deform_conv2d(input=x,
#                                  offset=offset,
#                                  weight=self.weight,
#                                  bias=self.bias,
#                                  padding=self.padding,
#                                  mask=None,
#                                  stride=self.stride,
#                                  dilation=self.dilation)
#                 return x
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
# 注意：这里我们不再需要 try-except 来处理 tvdcn 和 mmcv

print("Using torchvision for deformable convolution (stable, no MetaTensor issue)")

class DeformConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        # 初始化父类 nn.Conv2d
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        # 定义偏移量预测卷积层
        # 输出通道数 = 2 * groups * kernel_h * kernel_w
        # 在没有指定 offset_groups 的情况下，groups=1, offset_groups=1，所以是 2 * kernel_size^2
        offset_channels = 2 * self.kernel_size[0] * self.kernel_size[0]
        
        self.offset_conv = nn.Conv2d(self.in_channels,
                                     offset_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding)

        # 初始化：保证起始时偏移量为零
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def forward(self, x):
        # 强制转换为 Tensor 以避免 MONAI MetaTensor 兼容性问题
        if hasattr(x, "as_tensor"):
            x = x.as_tensor() 
        
        # 1. 计算偏移量
        offset = self.offset_conv(x)
        
        # 2. 执行可变形卷积（直接调用 torchvision 函数）
        x = deform_conv2d(input=x,
                          offset=offset,
                          weight=self.weight,
                          bias=self.bias,
                          padding=self.padding,
                          mask=None,
                          stride=self.stride,
                          dilation=self.dilation)
        return x

# 如果您的代码中还有 DeformConv3d，需要单独处理
# 目前 torchvision 没有 DeformConv3d 的原生实现，您可能需要保留 tvdcn 或 mmcv 的 3D 实现
# ...
if __name__ == "__main__":
    from config import get_config

    image = torch.randn(1, 3, 224, 224).cuda()
    # dim=[64,128,256,512],num_heads=[2,4,8,16],kernel_size=[7,7,7,7],
    # net = DeformConv2d(3,32,3,2,1)
    # model = net.cuda()
    # out = model(image)
    # print(out.shape)
    # out1 = model(out)
    # print(out1.shape)
    net = DeformConv2d(3,32,7,4,2)
    model = net.cuda()
    out1 = model(image)
    CONV = nn.Conv2d(32,64,1).cuda()
    OUT1 = CONV(out1)
    print(out1.shape)
    print(OUT1.shape)
