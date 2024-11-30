import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from skvideo import measure
import pyfvvdp
import lpips


def gaussian_window(size, sigma, device):
    """
    Create a Gaussian window used for SSIM calculation.
    """
    # Create a tensor for the range
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel, device):
    """
    Create a 2D Gaussian window.
    """
    _1D_window = gaussian_window(window_size, sigma, device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window_sigma=1.5, size_average=True):
    """
    Calculate the SSIM (Structural Similarity Index) for a batch of images.
    """
    device = img1.device
    channel = img1.size(1)
    window = create_window(window_size, window_sigma, channel, device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


def sequence_ssim(views1, views2, ssim_range=(-1, 1), window_size=11, window_sigma=1.5, size_average=True):
    """
    Calculate the SSIM for a batch of sequences.
    sequence1, sequence2: tensors of shape [batch_size, num_views, channels, height, width]
    """
    batch_size, num_views, channels, height, width = views1.size()
    ssim_values = torch.zeros(batch_size, num_views).to(views1.device)

    for i in range(num_views):
        view1 = views1[:, i, :, :, :]
        view2 = views2[:, i, :, :, :]
        ssim_values[:, i] = ssim(view1, view2, window_size, window_sigma, size_average=False)
    
    # print(ssim_values)
    ssim_values = scale_ssim(ssim_values, ssim_range)

    return ssim_values.mean(dim=1) if size_average else ssim_values



def scale_ssim(value, ssim_range, min_ssim=-1, max_ssim=1):

    # Normalize SSIM to [0, 1]
    normalized_ssim = (value - min_ssim) / (max_ssim - min_ssim)
    # Scale to [-4, 1]
    value_range = ssim_range[1] - ssim_range[0]
    scaled_ssim = normalized_ssim * value_range - (value_range-1)

    scaled_ssim = torch.clamp(scaled_ssim, min=-1, max=1)
    return scaled_ssim


def scale_psnr(psnr, min_psnr, max_psnr):
    """
    Scale PSNR values from [min_psnr, max_psnr] to [-1, 1].

    Args:
    - psnr (torch.Tensor): PSNR values.
    - min_psnr (float): Minimum expected PSNR value.
    - max_psnr (float): Maximum expected PSNR value.

    Returns:
    - torch.Tensor: Scaled PSNR values.
    """
    # Normalize PSNR to [0, 1]
    normalized_psnr = (psnr - min_psnr) / (max_psnr - min_psnr)
    # Scale to [-1, 1]
    scaled_psnr = normalized_psnr * 2 - 1
    return scaled_psnr

def psnr_metric(batch1, batch2, max_pixel=1.0, min_psnr=0, max_psnr=50):
    """
    Calculate and scale the PSNR for batches of image sequences.

    Args:
    - batch1 (torch.Tensor): Batch of image sequences.
    - batch2 (torch.Tensor): Batch of image sequences to compare against.
    - max_pixel (float): Maximum possible pixel value in the images.
    - min_psnr (float): Minimum expected PSNR value for scaling.
    - max_psnr (float): Maximum expected PSNR value for scaling.

    Returns:
    - torch.Tensor: Scaled PSNR values for each sequence in the batch.
    """
    if batch1.shape != batch2.shape:
        raise ValueError("Input tensors must have the same dimensions.")

    mse = F.mse_loss(batch1, batch2, reduction='none')
    mse = mse.mean(dim=[2, 3, 4])
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))



    # Check for PSNR values outside the expected range
    # violated_psnr = psnr[(psnr < min_psnr) | (psnr > max_psnr)]
    # if len(violated_psnr) > 0:
    #     raise ValueError(f"PSNR values outside the expected range [{min_psnr}, {max_psnr}]: {violated_psnr}")

    psnr = torch.clamp(psnr, min=min_psnr, max=max_psnr)
    

    # Scale PSNR values
    scaled_psnr = scale_psnr(psnr, min_psnr, max_psnr)

    return scaled_psnr.mean(dim=1)



def rgb_to_lum(images):
    """
    Convert a batch of RGB images to luminance (LUM).

    Parameters:
    - images: A PyTorch tensor of shape [batch_size, num_views, channels, height, width]
              with `channels` = 3 for RGB.

    Returns:
    - A PyTorch tensor of luminance images of shape [batch_size, num_views, 1, height, width].
    """
    # Ensure the input tensor is of float type for multiplication
    if images.dtype != torch.float32:
        images = images.to(dtype=torch.float32)

    # Define the weights for the RGB channels
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32, device=images.device)
    
    # Reshape weights to match the input dimensions: [1, 1, 3, 1, 1] for broadcast multiplication
    weights = weights.view(1, 1, 3, 1, 1)
    
    # Convert RGB to Luminance (LUM) by applying the weights
    lum_images = torch.sum(images * weights, dim=2, keepdim=True)
    
    return lum_images



def scale_strred(value, min_strred=0, max_strred=15):

    value = torch.clamp(value, min=min_strred, max=max_strred)

    normalized_strred = (value-min_strred) / (max_strred-min_strred)
    scaled_strred = 2 * normalized_strred - 1
    scaled_strred = - scaled_strred

    return scaled_strred


def strred(batch1, batch2):
    device = batch1.device
    batch_size, num_views, channels, height, width = batch1.size()
    
    if channels == 3:
        batch1 = rgb_to_lum(batch1)
        batch2 = rgb_to_lum(batch2)
    
    preds = []
    for seq1, seq2 in zip(batch1, batch2):
        seq1 = seq1.permute(0, 2, 3, 1).cpu().numpy()  # Shape: [num_views, channels, height, width]
        seq2 = seq2.permute(0, 2, 3, 1).cpu().numpy()  # Same shape as seq1
        
        pred = measure.strred(seq1, seq2)[1]  # This operation is done on CPU with NumPy arrays
        preds.append(pred)
    
    # Convert the list of predictions back to a PyTorch tensor and move to the original device
    preds_tensor = torch.tensor(preds, dtype=torch.float32).to(device)

    # print(preds_tensor)

    preds_tensor = scale_strred(preds_tensor)
    
    return preds_tensor




def scale_vdp(value, min_vdp=0, max_vdp=10):

    value = torch.clamp(value, min=min_vdp, max=max_vdp)

    normalized_vdp = (value-min_vdp) / (max_vdp-min_vdp)
    scaled_vdp = 2 * normalized_vdp - 1
    # scaled_strred = - scaled_strred

    return scaled_vdp



def fov_video_vdp(batch1, batch2):

    device = batch1.device
    batch_size, num_views, channels, height, width = batch1.size()

    fv = pyfvvdp.fvvdp(display_name='standard_fhd', heatmap=None, device=device)
    
    # if channels == 3:
    #     batch1 = rgb_to_lum(batch1)
    #     batch2 = rgb_to_lum(batch2)
    
    preds = []
    for seq1, seq2 in zip(batch1, batch2):
        # seq1 = seq1.permute(0, 2, 3, 1).cpu().numpy()  # Shape: [num_views, channels, height, width]
        # seq2 = seq2.permute(0, 2, 3, 1).cpu().numpy()  # Same shape as seq1
        
        # pred = measure.strred(seq1, seq2)[1]  # This operation is done on CPU with NumPy arrays

        # seq1 = seq1.permute(0, 2, 3, 1)  # Shape: [num_views, channels, height, width]
        # seq2 = seq2.permute(0, 2, 3, 1)  # Same shape as seq1

        pred, _ = fv.predict( seq2, seq1, dim_order="FCHW", frames_per_second=30 )


        preds.append(pred)
    
    # Convert the list of predictions back to a PyTorch tensor and move to the original device
    preds_tensor = torch.tensor(preds, dtype=torch.float32).to(device)

    # print(preds_tensor)

    # preds_tensor = scale_strred(preds_tensor)
    preds_tensor = scale_vdp(preds_tensor)
    
    return preds_tensor





class LPIPS_Seq(lpips.LPIPS):
    def forward(self, images1, images2):
        # images1, images2: [B, n_seq, C, H, W]
        B, n_seq, C, H, W = images1.size()
        lpips_scores = []

        # Use torch.no_grad() to prevent gradient calculation
        with torch.no_grad():
            for i in range(n_seq):
                # Select the ith frame from each sequence
                img1 = images1[:, i, :, :, :]  # [B, C, H, W]
                img2 = images2[:, i, :, :, :]  # [B, C, H, W]

                # Compute LPIPS score for the ith frame pair
                score = super().forward(img1, img2)  # [B, 1, 1, 1]
                lpips_scores.append(score)

        # Stack scores and compute the mean
        lpips_scores = torch.stack(lpips_scores, dim=1)  # [B, n_seq, 1, 1, 1]
        lpips_avg_scores = lpips_scores.mean(dim=1)  # [B, 1, 1, 1]

        lpips_avg_scores = 2 * lpips_avg_scores - 1 # [-1, 1]

        # lpips_avg_scores = - lpips_avg_scores # [1, -1]

        return lpips_avg_scores.squeeze()  # [B]

# Example usage:
# lpips_seq = LPIPS_Seq(net='vgg')
# loss = lpips_seq(image_seq1, image_seq2)



def est_params(frame, blk, sigma_nn):
    h, w = frame.shape
    sizeim = (torch.floor(torch.tensor(frame.shape, dtype=torch.float32) / blk) * blk).int()

    frame = frame[:sizeim[0], :sizeim[1]]

    # paired_products
    temp = []
    for u in range(blk):
        for v in range(blk):
            temp.append(frame[v:(sizeim[0] - (blk - v) + 1), u:(sizeim[1] - (blk - u) + 1)].reshape(-1))
    temp = torch.stack(temp, dim=0).float()

    cov_mat = torch.cov(temp).float()

    # force PSD
    eigval, eigvec = torch.linalg.eig(cov_mat)
    eigval = eigval.float()
    eigvec = eigvec.float()
    Q = eigvec
    xdiag = torch.diag(torch.maximum(eigval, torch.tensor(0.0)))
    cov_mat = Q @ xdiag @ Q.T

    temp = []
    for u in range(blk):
        for v in range(blk):
            temp.append(frame[v::blk, u::blk].reshape(-1))
    temp = torch.stack(temp, dim=0).float()

    # float32 vs float64 difference between python2 and python3 
    # avoiding this problem with quick cast to float64
    V, d = torch.linalg.eigh(cov_mat.double())
    V = V.float()

    # Estimate local variance
    sizeim_reduced = (sizeim / blk).int()
    ss = torch.zeros((sizeim_reduced[0], sizeim_reduced[1]), dtype=torch.float32)
    if torch.max(V) > 0:
        # avoid the matrix inverse for extra speed/accuracy
        ss = torch.linalg.solve(cov_mat, temp)
        ss = torch.sum((ss * temp) / (blk ** 2), dim=0)
        ss = ss.reshape(sizeim_reduced)

    V = V[V > 0]

    # Compute entropy
    ent = torch.zeros_like(ss, dtype=torch.float32)
    for u in range(V.shape[0]):
        ent += torch.log2(ss * V[u] + sigma_nn) + torch.log(2 * np.pi * np.exp(1))

    return ss, ent


def extract_info(frame1, frame2):
    blk = 3
    sigma_nsq = 0.1
    sigma_nsqt = 0.1

    model = SpatialSteerablePyramid(height=6)
    y1 = model.extractSingleBand(frame1, filtfile="sp5Filters", band=0, level=4)
    y2 = model.extractSingleBand(frame2, filtfile="sp5Filters", band=0, level=4)

    ydiff = y1 - y2

    ss, q = est_params(y1, blk, sigma_nsq)
    ssdiff, qdiff = est_params(ydiff, blk, sigma_nsqt)

    spatial = q * torch.log2(1 + ss)
    temporal = qdiff * torch.log2(1 + ss) * torch.log2(1 + ssdiff)

    return spatial, temporal


def strred_torch(referenceVideoData, distortedVideoData):
    referenceVideoData = torch.tensor(referenceVideoData, dtype=torch.float32)
    distortedVideoData = torch.tensor(distortedVideoData, dtype=torch.float32)

    assert referenceVideoData.shape == distortedVideoData.shape

    T, M, N, C = referenceVideoData.shape

    assert C == 1, f"strred called with videos containing {C} channels. Please supply only the luminance channel"

    referenceVideoData = referenceVideoData[:, :, :, 0]
    distortedVideoData = distortedVideoData[:, :, :, 0]

    rreds = []
    rredt = []

    rredssn = []
    rredtsn = []

    for i in range(0, T - 1, 2):
        refFrame1 = referenceVideoData[i]
        refFrame2 = referenceVideoData[i + 1]

        disFrame1 = distortedVideoData[i]
        disFrame2 = distortedVideoData[i + 1]

        spatialRef, temporalRef = extract_info(refFrame1, refFrame2)
        spatialDis, temporalDis = extract_info(disFrame1, disFrame2)

        rreds.append(torch.mean(torch.abs(spatialRef - spatialDis)).item())
        rredt.append(torch.mean(torch.abs(temporalRef - temporalDis)).item())

        rredssn.append(torch.abs(torch.mean(spatialRef - spatialDis)).item())
        rredtsn.append(torch.abs(torch.mean(temporalRef - temporalDis)).item())

    rreds = torch.tensor(rreds)
    rredt = torch.tensor(rredt)
    rredssn = torch.tensor(rredssn)
    rredtsn = torch.tensor(rredtsn)

    srred = torch.mean(rreds)
    trred = torch.mean(rredt)
    srredsn = torch.mean(rredssn)
    trredsn = torch.mean(rredtsn)

    strred = srred * trred
    strredsn = srredsn * trredsn

    return torch.stack((rreds, rredt, rredssn, rredtsn), dim=1), strred.item(), strredsn.item()



