import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, LSTM
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as transforms

from . import encoders
import json
import os
from PIL import Image
import cv2
import numpy as np





def img_to_lum_np(img):
    lum = img.convert('L')
    np_img = np.array(lum)
    np_img = np.expand_dims(np_img, axis=-1)
    return np_img

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




class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.attention_weights = nn.Linear(feature_dim, 1)

    def forward(self, features):
        # features shape: (seq_len, batch_size, feature_dim)
        attn_scores = self.attention_weights(features).softmax(dim=0)
        weighted_features = features * attn_scores
        return weighted_features.sum(dim=0)


class EfficientSequenceModel(nn.Module):
    def __init__(self, model_type='TransEncoder', enc_type="efficient", img_mode="RGB", view_enc_size=128, with_low_level=None, view_pretrained=False,angular_feat_size=1024, fusion="attention", num_layers=3, nhead=8, version='v0.0.0'):
        super(EfficientSequenceModel, self).__init__()
        self.view_enc_size = view_enc_size
        self.with_low_level = with_low_level
        self.view_pretrained = view_pretrained
        self.angular_feat_size = angular_feat_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.model_type = model_type
        self.enc_type = enc_type
        self.img_mode = img_mode
        self.version = version

        if self.with_low_level != None:
            view_enc_size = view_enc_size + sum(self.with_low_level)
        
        self.final_view_enc_size = view_enc_size
        

        

        # 
        self.fusion = fusion

        if fusion == "attention":
            self.angular_feat_fuser = AttentionFusion(self.final_view_enc_size)
        

        self.model_name = self.get_model_name()


        # CNN-based view encoder
        # https://pytorch.org/vision/stable/models.html
        # self.view_encoder = models.efficientnet.EfficientNet( weights='IMAGENET1K_V1', pretrained=self.view_pretrained)
        # self.view_encoder.fc = nn.Linear(self.view_encoder.fc.in_features, self.final_view_enc_size)

        self.view_encoder = encoders.ViewEncoder(model_type=self.enc_type, img_mode=img_mode, pred_scores=False, n_out_feats=self.view_enc_size, with_low_level=self.with_low_level, version=version)


        
        
        self.div_term = torch.exp(torch.arange(0, self.final_view_enc_size, 2) * -(torch.log(torch.tensor(10000.0)) / self.final_view_enc_size))

        # Transformer or LSTM
        if model_type == 'TransEncoder':
            layer = TransformerEncoderLayer(d_model=self.final_view_enc_size, nhead=nhead)
            self.angular_encoder = TransformerEncoder(layer, num_layers=num_layers)

        
        elif 'TransDecoder' in model_type:
            layer = TransformerDecoderLayer(d_model=self.final_view_enc_size, nhead=nhead)
            self.angular_encoder = TransformerDecoder(layer, num_layers=num_layers)
        elif model_type == 'LSTM':
            self.angular_encoder = LSTM(self.final_view_enc_size, self.final_view_enc_size, num_layers=num_layers, batch_first=False)

        # Final layer to output angular feature
        self.fc = nn.Linear(self.final_view_enc_size, angular_feat_size)
    

    def fuse_angular_features(self, angular_features):
        if self.fusion == "attention":
            final_features = self.angular_feat_fuser(angular_features)
        elif self.fusion == "mean":
            final_features = angular_features.mean(dim=0)
        elif self.fusion == "max":
            final_features = angular_features.max(dim=0)[0]
        elif self.fusion == "last":
            final_features = angular_features[-1, :, :]
        

        return final_features


    def generate_positional_encodings(self, seq_length, feature_size, device):
        """
        Generate positional encodings for a given sequence length and feature size.
        """
        position = torch.arange(seq_length).unsqueeze(1).to(device)
        
        if device != self.div_term: self.div_term = self.div_term.to(device)
        pe = torch.zeros(seq_length, feature_size).to(device)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        return pe
    


    def forward(self, x):

        device = x.device
        # assume the input is in shape of (batch_size, num_views, C, H, W)
        batch_size, num_views, C, H, W = x.size()

        # reshape to (num_views, batch_size, C, H, W)
        x = x.swapaxes(0, 1)

        # Process views individually to save memory
        if self.with_low_level == None:
            view_features = [self.view_encoder(x[i, :, :, :, :]).unsqueeze(0) for i in range(num_views)]
        else:
            view_features = [self.view_encoder(x[i, :, :, :, :]) for i in range(num_views)]

            view_features = [torch.cat(vf, dim=1).unsqueeze(0) for vf in view_features]

        
        view_features = torch.cat(view_features, dim=0)

        # print(view_features.shape)


        # Apply Transformer or LSTM with gradient checkpointing
        if 'TransDecoder' == self.model_type:
            angular_features = checkpoint(self.angular_encoder, view_features, view_features)

        elif self.model_type == 'TransEncoder':


            pos_encoding = self.generate_positional_encodings(num_views, self.final_view_enc_size, device)


            # pos_encoding = pos_encoding.to(device)

            # print(view_features.shape, pos_encoding.shape)
            view_features = view_features + pos_encoding.unsqueeze(1).expand(-1, batch_size, -1)

            angular_features = checkpoint(self.angular_encoder, view_features)

            

        elif self.model_type == 'LSTM':
            angular_features, _ = checkpoint(self.angular_encoder, view_features)

            


        final_features = self.fuse_angular_features(angular_features)

        
        # angular_features = angular_features.mean(dim=1)
        output = self.fc(final_features)

        return output



    def predict_features_from_view_paths(self, view_paths, device, batch_size=10):
        """
        Predict features from file paths of views of a sequence.

        Args:
        - view_paths (list of str): File paths of views.
        - device (torch.device): The device to perform computations on.

        Returns:
        - Tensor: The predicted features.
        """
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        

        all_view_features = []

        # Process views in batches
        for i in range(0, len(view_paths), batch_size):
            batch_paths = view_paths[i:i + batch_size]
            # img_tensors = [transform(Image.open(path)).to(device) for path in batch_paths]

            img_tensors = [img_path_to_tensor(path, self.img_mode).to(device) for path in batch_paths]

            # img_tensors = [transform(open_image_convert_to_rgb(path)) for path in batch_paths]
            img_tensors = torch.stack(img_tensors)
            img_tensors.to(device)

            # Generate view features for the batch
            with torch.no_grad():
                batch_view_features = self.view_encoder(img_tensors)

                if self.with_low_level != None:
                    batch_view_features = torch.concat(batch_view_features, dim=1)

                all_view_features.append(batch_view_features)

        # Concatenate all batch view features
        all_view_features = torch.cat(all_view_features, dim=0)

        all_view_features = all_view_features.unsqueeze(1)

        # print("all_view_features", all_view_features.shape)
        # Generate final output features using angular_encoder
        with torch.no_grad():

            if self.model_type == "TransEncoder":
                num_views = all_view_features.shape[0]
                # pos_encoding = self.generate_positional_encodings(num_views, self.view_enc_size, device)


                pos_encoding = self.generate_positional_encodings(num_views, self.final_view_enc_size, device)


                # pos_encoding = pos_encoding.to(device)

                # print(view_features.shape, pos_encoding.shape)
                all_view_features = all_view_features + pos_encoding.unsqueeze(1).expand(-1, 1, -1)

            # print("all_view_features", all_view_features.shape)
            angular_features = self.angular_encoder(all_view_features)

            if self.model_type == "LSTM":
                angular_features, _ = angular_features

            # print("angular_features", angular_features.shape)



            final_features = self.fuse_angular_features(angular_features)

            
            # print("final_features", final_features.shape)

            output_features = self.fc(final_features.squeeze(1))

        
        # print("output_features", output_features.shape)
        output_features = output_features.squeeze(0)
        # print("output_features", output_features.shape)
        return output_features
    


    def predict_features_from_patches(self, view_paths, device, patch_size=224, pooling='average'):
        """
        Predict features by processing patches from the same position across frames through the angular_encoder.

        Args:
        - view_paths (list of str): File paths of frames in a video.
        - device (torch.device): The device to perform computations on.
        - patch_size (int): Size of each square patch.
        - pooling (str): Type of pooling to fuse patch features ('average' or 'max').

        Returns:
        - Tensor: The fused predicted features.
        """
        transform = transforms.ToTensor()

        # Load and transform all frames
        frames = [transform(Image.open(path)).unsqueeze(0).to(device) for path in view_paths]
        
        # frames = [transform(open_image_convert_to_rgb(path)).unsqueeze(0).to(device) for path in view_paths]

        # Stack frames into a single tensor for efficient processing
        video_tensor = torch.cat(frames, dim=0)  # Shape: [num_frames, C, H, W]

        # Calculate the number of patches
        _, _, H, W = video_tensor.shape
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        all_features = []

        # Process each patch position across all frames
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Extract the patch across all frames
                patch = video_tensor[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

                # print(patch.shape)

                # Process the patch through the view_encoder and angular_encoder
                with torch.no_grad():
                    patch_view_features = self.view_encoder(patch)
                    if self.with_low_level is not None:
                        patch_view_features = torch.concat(patch_view_features, dim=1)
                    # print(patch_view_features.shape)
                    patch_angular_features = self.angular_encoder(patch_view_features.unsqueeze(1))
                    if self.model_type == "LSTM":
                        patch_angular_features, _ = patch_angular_features
                

                patch_angular_features = self.fuse_angular_features(patch_angular_features)

                

                all_features.append(patch_angular_features)

        # Concatenate features from all patches
        all_features = torch.cat(all_features, dim=0)

        # Pooling over patches
        if pooling == 'average':
            pooled_features = torch.mean(all_features, dim=0, keepdim=True)
        elif pooling == 'max':
            pooled_features, _ = torch.max(all_features, dim=0, keepdim=True)
        else:
            raise ValueError("Unsupported pooling type. Use 'average' or 'max'.")

        # Final processing to get the output features
        # with torch.no_grad():
        #     output_features = self.fc(pooled_features.squeeze(1))
        with torch.no_grad():
            output_features = self.fc(pooled_features.squeeze(1))
        output_features = output_features.squeeze(0)
        return output_features




    def get_model_name(self):

        if self.with_low_level == None:
            low_level_str = "0"
        else:
            with_low_level_str = [str(wll) for wll in self.with_low_level]
            low_level_str = '+'.join(with_low_level_str)

        model_name = f"SeqModel_{self.model_type}_{self.img_mode.lower()}_viewsz({self.view_enc_size})_lowfeat({low_level_str})_fviewsz({self.final_view_enc_size})_pretr({int(self.view_pretrained)})_fuse({self.fusion})angsz({self.angular_feat_size})_nlyr({self.num_layers})_{self.version}"

        return model_name
    


    def save_config(self, save_dir='model_configs'):
        
        config = {
            'model_type': self.model_type,
            'img_mode': self.img_mode,
            'enc_type': self.enc_type,
            'view_enc_size': self.view_enc_size,
            "with_low_level": self.with_low_level,
            "view_pretrained": self.view_pretrained,
            'angular_feat_size': self.angular_feat_size,
            "fusion": self.fusion,
            'num_layers': self.num_layers,
            'nhead': self.nhead,
            'version': self.version,
            # "final_view_enc_size": self.final_view_enc_size
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
    # Example usage for saving config
    model = EfficientSequenceModel(view_enc_size=512, angular_feat_size=256, num_layers=3, nhead=8, model_type='TransDecoder', version='1.0')
    model_name = model.save_config()
    print(f"Model configuration saved. Model name: {model_name}")

    # Example usage for loading from config
    config_path = 'model_configs/TransDecoder_v1.0_config.json'
    loaded_model = EfficientSequenceModel.load_from_config(config_path)
    print(f"Model loaded from config. Model type: {loaded_model.model_type}, Version: {loaded_model.version}")