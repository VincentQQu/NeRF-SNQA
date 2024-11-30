import torch
import torch.nn as nn
import torch.nn.functional as F
import json, os
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image





def img_path_to_tensor(img_path, img_mode="RGB"):
    image = Image.open(img_path)

    if img_mode == "LUM":
        image = image.convert('L')
        np_img = np.array(image)
        np_img = np.expand_dims(np_img, axis=-1)
    else:
        np_img = np.array(image)

    # Check if the image has three dimensions (H, W, C)
    # if np_img.ndim == 3:
    #     # Change the dimension order from HWC to CHW
    #     np_img = np_img.transpose((2, 0, 1))

    np_img = np.transpose(np_img, (2, 0, 1))
    image_tensor = torch.from_numpy(np_img).float()  # Convert to float tensor
    image_tensor /= 255.0  # Normalize to the range [0, 1]

    return image_tensor

def default_norm(out_channels):
    # return nn.GroupNorm(1, out_channels)
    return nn.LayerNorm(out_channels)
    # return nn.BatchNorm3d(out_channels) 


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


class R2Plus1D_Block_LN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(R2Plus1D_Block_LN, self).__init__()
        # Spatial convolution
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, (kernel_size[0], kernel_size[1], 1), stride=(stride, stride, 1), padding=(padding, padding, 0), bias=False)
        # LayerNorm does not assume fixed input size; normalization is applied across the channel dimension
        # self.ln1 = nn.LayerNorm([1, 1, out_channels])
        self.ln1 = default_norm(out_channels)

        # Temporal convolution
        self.temporal_conv = nn.Conv3d(out_channels, out_channels, (1, 1, kernel_size[2]), stride=(1, 1, stride), padding=(0, 0, padding), bias=False)
        # self.ln2 = default_norm([1, 1, out_channels])

        self.ln2 = default_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = x
        x = self.spatial_conv(x)
        # Permute + LayerNorm + Permute back
        x = self.ln1(x.permute(0, 2, 3, 4, 1).contiguous()).permute(0, 4, 1, 2, 3)

        x = self.relu(x)
        
        x = self.temporal_conv(x)
        x = self.ln2(x.permute(0, 2, 3, 4, 1).contiguous()).permute(0, 4, 1, 2, 3)

        if x0.shape == x.shape:
            x += x0

        x = self.relu(x)
        return x




# seq_model_config = {
#     "model_type": "TransEncoder", # LSTM, TransEncoder
#     "enc_type": "efficient", # normal, efficient
#     "view_enc_size": 256,
#     "with_low_level": [32, 64, 128],
#     "view_pretrained": False,
#     "angular_feat_size": 768,
#     "fusion": "attention", # attention, last, mean, max
#     "num_layers": 6,
#     "nhead": 8,
#     "version": version
# }


class ResNet_R2Plus1D_LN(nn.Module):
    def __init__(self, img_mode, chs=[6, 24, 64, 128, 256], angular_feat_size=512, version="v0.0.0"):
        super(ResNet_R2Plus1D_LN, self).__init__()

        self.img_mode = img_mode
        self.chs = chs
        self.angular_feat_size = angular_feat_size
        self.version = version

        self.model_name = self.get_model_name()

        self.in_channels = chs[0]

        img_ch = 3 if img_mode == "RGB" else 1
        self.conv1 = nn.Conv3d(img_ch, self.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=True) # , padding=(1, 3, 3)
        self.ln1 = default_norm([self.in_channels])  # Adjusted for variable input size
        self.relu = nn.ReLU(inplace=True)

        self.maxpool_convs = []

        for i in range(1, len(chs)):
            self.maxpool_convs.append(self._make_maxpool_conv(chs[i], stride=1, blocks=3))


        self.maxpool_convs = nn.Sequential(*self.maxpool_convs)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(chs[-1], angular_feat_size)


    def _make_maxpool_conv(self, out_channels, stride=1, blocks=3):
        maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2)) # , padding=(0, 1, 1)
        layer1 = self._make_layer(out_channels, stride=stride, blocks=blocks)

        return nn.Sequential(maxpool1, layer1)





    def _make_layer(self, out_channels, stride, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(R2Plus1D_Block_LN(self.in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride, padding=1))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Permute the input to match (batch_size, C, D, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # Adjusting from (batch_size, num_views, C, H, W) to (batch_size, C, num_views, H, W)
        
        # print("before conv1", x.shape)
        # Forward pass remains the same...
        x = self.conv1(x)

        
        # x = self.ln1(x.permute(0, 2, 3, 4, 1).contiguous()).permute(0, 4, 1, 2, 3)
        # x = self.relu(x)

        # print("after conv1", x.shape)


        x = self.maxpool_convs(x)
        # print("after maxpool_convs", x.shape)


        x = self.avgpool(x)
        # print("after avgpool", x.shape)

        x = torch.flatten(x, 1)

        # print("after flatten", x.shape)
        x = self.fc(x)

        # print(x)
        return x



    def predict_features_from_view_paths(self, view_paths, device, batch_size=10):
        """
        Predict features from file paths of views of a sequence.

        Args:
        - view_paths (list of str): File paths of views.
        - device (torch.device): The device to perform computations on.

        Returns:
        - Tensor: The predicted features.
        """
        transform = transforms.ToTensor()

        # Process views in batches
        # for i in range(0, len(view_paths), batch_size):
        #     batch_paths = view_paths[i:i + batch_size]
        #     img_tensors = [transform(open_image_convert_to_rgb(path)) for path in batch_paths]
        #     img_tensors = torch.stack(img_tensors).to(device)
        #     # Generate view features for the batch
        #     with torch.no_grad():
        #         self.forward(img_tensors)

        img_tensors = [img_path_to_tensor(path, img_mode=self.img_mode) for path in view_paths]

        # img_tensors = [transform(Image.open(path)) for path in view_paths]
        # img_tensors = [transform(open_image_convert_to_rgb(path)) for path in view_paths]

        # print(type(img_tensors))

        # print(device)

        img_tensors = torch.stack(img_tensors)
        # print(type(img_tensors))
        # print(img_tensors) 
        img_tensors = img_tensors.to(device)
        # print(img_tensors)
        # print(type(img_tensors))

        img_tensors = img_tensors.unsqueeze(0)

        # print(img_tensors.shape)

        # print(img_tensors)
        # Generate view features for the batch
        with torch.no_grad():
            output_features = self.forward(img_tensors)

        # print("output_features", output_features.shape)
        output_features = output_features.squeeze(0)
        # print("output_features", output_features.shape)
        return output_features

    def get_model_name(self):


        chs = [str(ch) for ch in self.chs]
        chs_str = '+'.join(chs)


        model_name = f"SeqModel_chs({chs_str})_angsz({self.angular_feat_size})_{self.version}"

        return model_name
    


    def save_config(self, save_dir='model_configs'):
        
        config = {
            "img_mode": self.img_mode, 
            'chs': self.chs,
            'angular_feat_size': self.angular_feat_size,

            'version': self.version,
        }

        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, f'{self.model_name}_config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        return self.model_name


    @classmethod
    def load_from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        return cls(**config)



if __name__ == "__main__":
    # Example usage
    # (batch_size, num_views, C, H, W)
    model = ResNet_R2Plus1D_LN(chs=[6, 24, 64, 128], angular_feat_size=512)
    # Example input with arbitrary sequence length and spatial dimensions
    input_tensor = torch.rand(16, 90, 3, 250, 250)  # batch_size=2, channels=3, depth=8, height=120, width=160
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Expected output shape: [2, 10]

    # 250*250*90*17