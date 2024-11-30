import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from . import qa_methods


class ContrastiveLearning(nn.Module):
    def __init__(self, encoder, projector, ssim_range=(-1, 1), psnr_range=(0, 50), loss_weights=[1, 0, 0, 0, 0, 0], tolerance=0):
        super(ContrastiveLearning, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.ssim_range = ssim_range
        self.psnr_range = psnr_range
        self.loss_weights = loss_weights
        self.tolerance = tolerance
        self.loss_fn_vgg = qa_methods.LPIPS_Seq(net='vgg')

    def forward(self, x1, x2):
        feat_1 = self.encoder(x1)
        feat_2 = self.encoder(x2)

        zs = [(proj(feat_1), proj(feat_2)) if weight != 0 else (torch.tensor(0), torch.tensor(0))
              for proj, weight in zip(self.projector.mlps, self.loss_weights)]
        return zs

    def compute_loss_part(self, z1, z2, x1, x2, metric_name, replace_ratios=None):
        if self.loss_weights[metric_name] == 0:
            return torch.tensor(0)

        cos_sim = F.cosine_similarity(z1, z2, dim=1, eps=1e-8)
        metric_fn_mapping = {
            0: lambda: qa_methods.sequence_ssim(x1, x2, ssim_range=self.ssim_range),
            1: lambda: qa_methods.psnr_metric(x1, x2, max_pixel=1.0, min_psnr=self.psnr_range[0], max_psnr=self.psnr_range[1]),
            2: lambda: replace_ratio_to_similarity(replace_ratios),
            3: lambda: qa_methods.strred(x1, x2),
            4: lambda: qa_methods.fov_video_vdp(x1, x2),
            5: lambda: self.loss_fn_vgg(x1, x2),
        }

        metric_values = metric_fn_mapping[metric_name]()

        print(f"{metric_values.mean():.4f}, {metric_values.std():.4f})==({cos_sim.mean():.4f}, {cos_sim.std():.4f})", end=";  ")

        return mae_with_tolerance(cos_sim, metric_values, tolerance=self.tolerance)

    def loss(self, x1, x2, replace_ratios):
        zs = self.forward(x1, x2)

        metric_names = ["diff_with_ssim", "diff_with_psnr", "loss_replace", "loss_strred", "loss_vdp", "loss_lpips"]
        total_loss = 0
        loss_values = {name: 0 for name in metric_names}

        for i, (z1, z2) in enumerate(zs):
            loss_part = self.compute_loss_part(z1, z2, x1, x2, i, replace_ratios)
            total_loss += self.loss_weights[i] * loss_part
            loss_values[metric_names[i]] = loss_part.item()

        loss_values["loss"] = total_loss.item()
        print()  # Print a new line after all debug output
        return total_loss, loss_values



class ContrastiveLearningAutoweight(nn.Module):
    def __init__(self, encoder, projector, ssim_range=(-1, 1), psnr_range=(0, 50), loss_weights=[1, 0, 0, 0, 0, 0], tolerance=0):
        super(ContrastiveLearningAutoweight, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.ssim_range = ssim_range
        self.psnr_range = psnr_range
        self.loss_weights = loss_weights
        self.tolerance = tolerance
        self.loss_fn_vgg = qa_methods.LPIPS_Seq(net='vgg')

        # self.log_vars = nn.Parameter(torch.zeros(len(loss_weights)))

        self.log_vars = nn.Parameter(torch.zeros(len(loss_weights)))

    def forward(self, x1, x2):
        feat_1 = self.encoder(x1)
        feat_2 = self.encoder(x2)

        zs = [(proj(feat_1), proj(feat_2)) if weight != 0 else (torch.tensor(0), torch.tensor(0))
              for proj, weight in zip(self.projector.mlps, self.loss_weights)]
        return zs

    def compute_loss_part(self, z1, z2, x1, x2, metric_name, replace_ratios=None):
        if self.loss_weights[metric_name] == 0:
            return torch.tensor(0)

        cos_sim = F.cosine_similarity(z1, z2, dim=1, eps=1e-8)
        metric_fn_mapping = {
            0: lambda: qa_methods.sequence_ssim(x1, x2, ssim_range=self.ssim_range),
            1: lambda: qa_methods.psnr_metric(x1, x2, max_pixel=1.0, min_psnr=self.psnr_range[0], max_psnr=self.psnr_range[1]),
            2: lambda: replace_ratio_to_similarity(replace_ratios),
            3: lambda: qa_methods.strred(x1, x2),
            4: lambda: qa_methods.fov_video_vdp(x1, x2),
            5: lambda: self.loss_fn_vgg(x1, x2),
        }

        metric_values = metric_fn_mapping[metric_name]()

        

        branch_loss = mae_with_tolerance(cos_sim, metric_values, tolerance=self.tolerance)**2

        # branch_loss = F.mse_loss(cos_sim, metric_values, reduction='mean')
        log_var = self.log_vars[metric_name]

        precision = torch.exp(-log_var)

        loss_part = 0.5 * precision * branch_loss + log_var


        print(f"({metric_values.mean():.4f}, {metric_values.std():.4f})==({cos_sim.mean():.4f}, {cos_sim.std():.4f})~log_var=[{log_var:.4f}]", end=";  ")

        return loss_part

    def loss(self, x1, x2, replace_ratios):
        zs = self.forward(x1, x2)

        metric_names = ["diff_with_ssim", "diff_with_psnr", "loss_replace", "loss_strred", "loss_vdp", "loss_lpips"]
        total_loss = 0
        loss_values = {name: 0 for name in metric_names}

        for i, (z1, z2) in enumerate(zs):
            loss_part = self.compute_loss_part(z1, z2, x1, x2, i, replace_ratios)
            total_loss += self.loss_weights[i] * loss_part
            loss_values[metric_names[i]] = loss_part.item()

        loss_values["loss"] = total_loss.item()
        print()  # Print a new line after all debug output
        return total_loss, loss_values








def replace_ratio_to_similarity(replace_ratios):
    """
    Convert replace ratios to similarity values.

    Args:
    - replace_ratios (np.array): Array of replace ratios, each ranging from 0 to 1.

    Returns:
    - np.array: Array of similarity values, each ranging from -1 to 1.
    """
    # Linear transformation: similarity = -2 * replace_ratio + 1
    similarity_values = -2 * replace_ratios + 1
    return similarity_values



class MLPs(nn.Module):
    def __init__(self, mlp_config, num_mlp=3):
        super(MLPs, self).__init__()
        self.mlp_config = mlp_config
        self.num_mlp = num_mlp

        self.mlps = nn.ModuleList([MLP(**mlp_config) for _ in range(num_mlp)])
        self.model_name = self.mlps[0].get_model_name()

    def save_config(self, save_dir='model_configs'):
        config = {
            "mlp_config": self.mlp_config,
            "num_mlp": self.num_mlp
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
        
        # print(config)
        return cls(config["mlp_config"], config["num_mlp"])

class MLP(nn.Module):

    def __init__(self, input_dim, mlp_dim, output_dim, version='v0.0.0'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.version = version

        self.model_name = self.get_model_name()

        mlp = [
            nn.Linear(input_dim, mlp_dim, bias=True),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, output_dim, bias=True)
        ]



        self.mlp = nn.Sequential(*mlp)



    def forward(self, x):
        return self.mlp(x)



    def get_model_name(self):

        model_name = f"Projector_MLP_in({self.input_dim})_mid({int(self.mlp_dim)})_out({self.output_dim})_{self.version}"

        return model_name
    


    def save_config(self, save_dir='model_configs'):
        
        config = {
            'input_dim': self.input_dim,
            'mlp_dim': self.mlp_dim,
            "output_dim": self.output_dim,
            "version": self.version,

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



class ContinuousAmplifyGradientLoss(nn.Module):
    def __init__(self, base_loss_fn=nn.L1Loss(), amplification_base=2.0, epsilon=1e-8):
        """
        Initializes the custom loss function with continuous gradient amplification.

        Parameters:
        - base_loss_fn: The base loss function (e.g., nn.MSELoss()).
        - amplification_base: The base for the amplification, determining its strength.
        - epsilon: A small value added for numerical stability.
        """
        super(ContinuousAmplifyGradientLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.amplification_base = amplification_base
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Forward pass for the loss function.

        Parameters:
        - input: The predictions from the model.
        - target: The ground truth labels/targets.
        """
        # Compute the base loss
        base_loss = self.base_loss_fn(input, target)

        # Compute gradients of the base loss w.r.t. inputs
        input_grad = torch.autograd.grad(base_loss, input, create_graph=True, only_inputs=True)[0]

        # Compute the magnitude of gradients
        grad_magnitude = torch.norm(input_grad) + self.epsilon

        # Calculate the amplification factor as inversely proportional to the gradient magnitude
        amplification_factor = 1 + self.amplification_base / grad_magnitude

        # Apply the amplification factor to the base loss
        amplified_loss = base_loss * amplification_factor

        return amplified_loss

# Example usage
# base_loss_fn = nn.MSELoss()
# custom_loss_fn = ContinuousAmplifyGradientLoss(base_loss_fn)
# loss = custom_loss_fn(predictions, targets)


def mae_with_tolerance(y_pred, y_true, tolerance=0.1):
    """
    Compute Mean Absolute Error with a tolerance level.

    Args:
    - y_pred (Tensor): Predicted values. Shape: (B, *)
    - y_true (Tensor): True values. Shape: (B, *)
    - tolerance (float): Tolerance level for MAE.

    Returns:
    - Tensor: MAE with tolerance applied. Shape: (B, *)
    """
    # # Calculate absolute error
    # abs_error = torch.abs(y_pred - y_true)

    # # Apply tolerance: Set MAE to zero where error is below tolerance
    # abs_error[abs_error < tolerance] = 0

    # # Calculate mean across dimensions except for the batch dimension
    # mae = torch.mean(abs_error, dim=tuple(range(1, abs_error.dim())))

    mae = nn.L1Loss()(y_pred, y_true)

    # mae = ContinuousAmplifyGradientLoss()(y_pred, y_true)

    return mae


def count_parameters(module: nn.Module, trainable: bool = True) -> int:

    if trainable:
        num_parameters = sum(p.numel() for p in module.parameters()
                             if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in module.parameters())

    return num_parameters