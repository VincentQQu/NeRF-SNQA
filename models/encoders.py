import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
from torchsummary import summary
# import math




# https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=4):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)




# https://towardsdatascience.com/residual-bottleneck-inverted-residual-linear-bottleneck-mbconv-explained-89d7b7e7c6bc
# https://pytorch.org/vision/main/_modules/torchvision/models/efficientnet.html
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, padding_mode, MBC_type="fused", expansion=4):

        expanded_features = in_channels * expansion
        super().__init__()

        if MBC_type == "depthwise":
            self.mbconv = nn.Sequential(
                # narrow -> wide
                ConvNormAct(in_channels, expanded_features, kernel_size=1),
                # wide -> wide
                ConvNormAct(expanded_features, expanded_features, kernel_size=kernel_size,
                padding=padding, padding_mode=padding_mode,  groups=expanded_features), # 
                # here you can apply SE
                SE_Block(expanded_features),
                # wide -> narrow
                ConvNormAct(expanded_features, out_channels, kernel_size=1),
            )
        elif MBC_type == "fused":
            self.mbconv = nn.Sequential(
                # narrow -> wide
                ConvNormAct(in_channels, expanded_features, kernel_size=1),
                # here you can apply SE
                SE_Block(expanded_features),
                # wide -> narrow
                ConvNormAct(expanded_features, out_channels, kernel_size=kernel_size,
                padding=padding, padding_mode=padding_mode, groups=1),
            )
    
    def forward(self, x):
        # print("++++++++++++++")
        x1 = x
        x2 = self.mbconv(x)
        # print(x2.shape)
        # if x1.shape == x2.shape: print("equ")
        # else: print("not equ")
        return x1 + x2 if x1.shape == x2.shape else x2






def default_norm(out_channels):
    # return nn.GroupNorm(1, out_channels)
    return nn.BatchNorm2d(out_channels) 


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', padding_mode="reflect", groups=1):
        # if act == None: nn.SiLU(inplace=True)
        super(ConvNormAct, self).__init__()
        self.convna = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding, padding_mode=padding_mode,
                groups=groups
            ),
            default_norm(out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        x1 = x
        x2 = self.convna(x)

        return x1 + x2 if x1.shape == x2.shape else x2
        # return self.convna(x)





class double_conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=[3,3], stride=[2,1] , padding='same', padding_mode="reflect", groups=1):
        # if act == None: nn.SiLU(inplace=True)
        super(double_conv, self).__init__()
        self.convna = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding, padding_mode=padding_mode,
                groups=groups
            ),
            default_norm(mid_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=kernel_size[1],
                stride=stride[1],
                padding=padding, padding_mode=padding_mode,
                groups=groups
            ),
            default_norm(out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        x1 = x
        x2 = self.convna(x)

        return x1 + x2 if x1.shape == x2.shape else x2
        # return self.convna(x)





def default_pool(strides):
    return nn.MaxPool2d(strides)
    # return nn.AvgPool2d(strides)


class ViewEncoder(nn.Module):
    def __init__(self, model_type="efficient", img_mode="RGB", pred_scores=False, n_out_feats=256, with_low_level=None, version="v0.0.0"):
        super(ViewEncoder, self).__init__()

        assert model_type in ["efficient", "normal"], "model_type should be either 'efficient' or 'normal'"

        self.model_type = model_type
        self.n_out_feats = n_out_feats
        self.with_low_level = with_low_level
        self.img_mode = img_mode
        self.version = version

        default_padding = "same" # valid, same
        default_padding_mode = "reflect" # circular, reflect, replicate, zeros



        if self.img_mode == "LUM":
            conv_0_chs = [1, 12, 12]
        else:
            conv_0_chs = [3, 6, 12]


        self.conv_0 = double_conv(in_channels=conv_0_chs[0], mid_channels=conv_0_chs[1], out_channels=conv_0_chs[2], kernel_size=[5,3], stride=[2,2] , padding=0, padding_mode=default_padding_mode, groups=1)
        
        # initial padding=0

        chs = [conv_0_chs[-1]* (2)**i for i in range(5)]
        self.chs = chs

        l_i = 0
        if self.with_low_level != None:
            self.low_level_convs = nn.ModuleList()
            for low_level_dim in self.with_low_level:
                conv_pool = nn.Sequential(
                    default_pool((2, 2)),
                    double_conv(in_channels=chs[l_i], mid_channels=low_level_dim, out_channels=low_level_dim, kernel_size=[3,3], stride=[1,1] , padding=default_padding, padding_mode=default_padding_mode, groups=1),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.low_level_convs.append(conv_pool)
                l_i += 1

        

        ############### eff convs ###############
        if model_type=="efficient":

            # [B, 1, 36, 120]
            self.fused_conv_1 = MBConv(in_channels=chs[0], out_channels=chs[1], kernel_size=(3, 3), padding=default_padding, padding_mode=default_padding_mode, MBC_type="fused", expansion=1)

            self.fused_conv_2 = MBConv(in_channels=chs[1], out_channels=chs[1], kernel_size=(3, 3), padding=default_padding, padding_mode=default_padding_mode, MBC_type="fused", expansion=1)

            # self.pool_1 = nn.MaxPool2d((2, 2))
            
            # [B, 4, 36, 60]
            self.fused_conv_3 = MBConv(in_channels=chs[1], out_channels=chs[2], kernel_size=(3, 3), padding=default_padding, padding_mode=default_padding_mode, MBC_type="fused", expansion=1)

            self.fused_conv_4 = MBConv(in_channels=chs[2], out_channels=chs[2], kernel_size=(3, 3), padding=default_padding, padding_mode=default_padding_mode, MBC_type="fused", expansion=2)

            # self.pool_2 = nn.MaxPool2d((3, 3))

            # [B, 8, 36, 20]
            self.dwise_conv_1 = MBConv(in_channels=chs[2], out_channels=chs[3], kernel_size=(3, 3), padding=default_padding, padding_mode=default_padding_mode, MBC_type="depthwise", expansion=2)

            self.dwise_conv_2 = MBConv(in_channels=chs[3], out_channels=chs[3], kernel_size=(3, 3), padding=default_padding, padding_mode=default_padding_mode, MBC_type="depthwise", expansion=2)

            # self.pool_3 = nn.MaxPool2d((5, 5))

            # [B, 16, 36, 4]
            # self.pwise_conv = ConvNormAct(chs[3], chs[4], (1, 1), padding=0, padding_mode=default_padding, groups=1)

            # pre_feats_size = chs[4]
        
        ############### simple convs ###############
        elif model_type=="normal":

            self.fused_conv_1 = double_conv(in_channels=chs[0], mid_channels=chs[1], out_channels=chs[1], kernel_size=[3,3], stride=[1,1] , padding=default_padding, padding_mode=default_padding_mode, groups=1)

            self.fused_conv_2 = double_conv(in_channels=chs[1], mid_channels=chs[1], out_channels=chs[1], kernel_size=[3,3], stride=[1,1] , padding=default_padding, padding_mode=default_padding_mode, groups=1)

            
            
            # [B, 4, 36, 60]
            self.fused_conv_3 = double_conv(in_channels=chs[1], mid_channels=chs[2], out_channels=chs[2], kernel_size=[3,3], stride=[1,1] , padding=default_padding, padding_mode=default_padding_mode, groups=1)

            self.fused_conv_4 = double_conv(in_channels=chs[2], mid_channels=chs[2], out_channels=chs[2], kernel_size=[3,3], stride=[1,1] , padding=default_padding, padding_mode=default_padding_mode, groups=1)

            

            # [B, 8, 36, 20]
            self.dwise_conv_1 = double_conv(in_channels=chs[2], mid_channels=chs[3], out_channels=chs[3], kernel_size=[3,3], stride=[1,1] , padding=default_padding, padding_mode=default_padding_mode, groups=1)

            self.dwise_conv_2 = double_conv(in_channels=chs[3], mid_channels=chs[3], out_channels=chs[3], kernel_size=[3,3], stride=[1,1] , padding=default_padding, padding_mode=default_padding_mode, groups=1)


        self.pool_1 = default_pool((2, 2))


        self.pool_2 = default_pool((2, 2))

        self.pool_3 = default_pool((2, 2))

        # [B, 16, 36, 4]
        self.pwise_conv = ConvNormAct(chs[3], chs[4], (1, 1), padding=0, padding_mode=default_padding_mode, groups=1)

        pre_feats_size = chs[4]


        # max along seq: torch.max(input, dim=-1, keepdim=True)
        # flatten: torch.flatten(x)
        # [B, 16, 36, 1]
        self.out_feats = nn.Linear(pre_feats_size, n_out_feats)


        # [N, 256, 1, 1]
        self.pred_scores = pred_scores
        if pred_scores:
            self.out_score = nn.Linear(n_out_feats, 1)
    


    
    def forward(self, model_input):
        x = model_input

        x = self.conv_0(x)

        if self.with_low_level != None:
            low_level_feats = []
            low_feat = self.low_level_convs[0](x)
            low_feat = low_feat.view(low_feat.size(0), -1)
            low_level_feats.append(low_feat)

        # if self.model_type == "efficient":
        x = self.fused_conv_1(x)
        x = self.fused_conv_2(x)
        x = self.pool_1(x)

        if self.with_low_level != None:
            low_feat = self.low_level_convs[1](x)
            low_feat = low_feat.view(low_feat.size(0), -1)
            low_level_feats.append(low_feat)

        # print(x.shape)


        x = self.fused_conv_3(x)
        x = self.fused_conv_4(x)
        x = self.pool_2(x)

        if self.with_low_level != None:
            low_feat = self.low_level_convs[2](x)
            low_feat = low_feat.view(low_feat.size(0), -1)
            low_level_feats.append(low_feat)

        # print(x.shape)


        x = self.dwise_conv_1(x)
        x = self.dwise_conv_2(x)
        x = self.pool_3(x)

        # print(x.shape)

        x = self.pwise_conv(x)
        

            
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # print(x.shape)

        x = x.view(x.size(0), -1)

        # x = torch.flatten(x, start_dim=1)

        x = self.out_feats(x)


        if self.pred_scores:
            model_output = self.out_score(x)
        else:
            

            if self.with_low_level != None:
                low_level_feats.append(x)
                model_output = low_level_feats
            else:
                model_output = x

        return model_output







