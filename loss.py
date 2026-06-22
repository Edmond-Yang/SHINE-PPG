import torch
import logging
import torch.nn.functional as F
from einops import rearrange

log = logging.getLogger(__name__)

# signal level
def rppg_loss(pred_signal, gt_signal):
    neg_pearson = 1 - _pearson_correlation(pred_signal, gt_signal)
    return neg_pearson 


def _pearson_correlation(x, y):
    vx = x - torch.mean(x, dim=1, keepdim=True)
    vy = y - torch.mean(y, dim=1, keepdim=True)
    cost = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)))
    return torch.mean(cost)


def frequency_loss(videos, lambda_spatial=1.0, lambda_temporal=1.0):
    spatial_loss = frequency_spatial_multiscale(videos)
    return lambda_spatial * spatial_loss


def frequency_spatial_multiscale(videos, 
                             
                              D0_illumination_scales=[0.7, 1.5, 2.5], 
                              D0_reflectance_scales=[0.7, 1.5, 2.5], 
                              illumination_weight=[0.05,0.2,0.75], 
                              reflectance_weight=[0.75,0.2,0.05],
                              lambda_I=1.0, 
                              lambda_R=1.0):

    _, v_illumination, v_reflectance, _ = videos
    b, c, t, h, w = v_reflectance.shape
    device = v_reflectance.device

    reflectance_batch = rearrange(v_reflectance, 'b c t h w -> (b t) c h w').contiguous()
    illumination_batch = rearrange(v_illumination, 'b c t h w -> (b t) c h w').contiguous()

    total_low_freq_from_reflectance = torch.tensor(0.0, device=device)
    total_high_freq_from_illumination = torch.tensor(0.0, device=device)

    # For reflectance
    for D0, weight in zip(D0_reflectance_scales, reflectance_weight):
        low_freq_energy, _ = _spatial_frequency_loss(reflectance_batch, D0=D0)
        total_low_freq_from_reflectance += weight * torch.mean(low_freq_energy)

    # For illumination
    for D0, weight in zip(D0_illumination_scales, illumination_weight):
        _, high_freq_energy = _spatial_frequency_loss(illumination_batch, D0=D0)
        total_high_freq_from_illumination += weight * torch.mean(high_freq_energy)

    # average
    avg_low_freq_from_reflectance = total_low_freq_from_reflectance / len(D0_reflectance_scales)
    avg_high_freq_from_illumination = total_high_freq_from_illumination / len(D0_illumination_scales)

    loss = lambda_R * avg_low_freq_from_reflectance + lambda_I * avg_high_freq_from_illumination

    return loss

def _spatial_frequency_loss(frame_batch, D0):
  
    device = frame_batch.device
    frame_batch_gray = frame_batch.mean(dim=1)  # Shape: [N, H, W]

    fft_frame = torch.fft.fft2(frame_batch_gray)
    fft_frame_shifted = torch.fft.fftshift(fft_frame, dim=(-2, -1))  # 在 H, W 維度上移位

    n, rows, cols = frame_batch_gray.shape
    crow, ccol = rows // 2, cols // 2

    u, v = torch.meshgrid(torch.arange(rows, device=device),
                          torch.arange(cols, device=device),
                          indexing='ij')

    D = torch.sqrt((u - crow) ** 2 + (v - ccol) ** 2)

    low_pass_mask = torch.exp(-(D ** 2) / (2 * (D0 ** 2)))
    high_pass_mask = 1 - low_pass_mask

    low_pass_fft = fft_frame_shifted * low_pass_mask
    high_pass_fft = fft_frame_shifted * high_pass_mask

    low_freq_energy = torch.mean(torch.abs(low_pass_fft), dim=(-2, -1))
    high_freq_energy = torch.mean(torch.abs(high_pass_fft), dim=(-2, -1))

    return low_freq_energy, high_freq_energy



def reconstruct_loss(videos, ):
    v_original, v_illumination, v_reflectance, v_specular = videos
    v_diffuse_est = v_original - v_specular

    log_original = torch.log(torch.clamp(v_diffuse_est, min=1e-3))
    log_illumination = torch.log(torch.clamp(v_illumination, min=1e-3))
    log_reflectance = torch.log(torch.clamp(v_reflectance, min=1e-3))
    reconstruction_loss = torch.sqrt(torch.mean((log_original - log_illumination - log_reflectance) ** 2))

    return reconstruction_loss

def reflect_loss(videos): 

    v_original, v_illumination, v_reflectance, v_specular = videos
    
    af_reflectance, bf_reflectance = v_reflectance.chunk(2, dim=0)
    reflectance_consistency_loss = torch.sqrt(torch.mean((af_reflectance - bf_reflectance) ** 2))
   
    return reflectance_consistency_loss


def _kl_loss(spec_mag, dark):

    q = spec_mag.flatten(1)
    p = dark.flatten(1)

    q = q.clamp(min=1e-6)
    p = p.clamp(min=1e-6)
    q = q / q.sum(dim=1, keepdim=True)
    p = p / p.sum(dim=1, keepdim=True)
    
    return torch.mean(torch.sum(q * torch.log(q / p), dim=1))

def specular_loss(videos, lambda_kl=2.0, lambda_consistency=2.0, lambda_sparse=0.7):
    """
    計算高光相關 Loss，包含稀疏性、分布約束與顏色一致性。
    """
    v_original, v_illumination, v_reflectance, v_specular = videos
    eps = 1e-6

    spec_mag = torch.norm(v_specular, dim=1, keepdim=True)
    dark, _ = torch.min(v_original, dim=1, keepdim=True)

    sparsity_loss = torch.mean(spec_mag)
    kl_loss = _kl_loss(spec_mag, dark)

    weight_cos = spec_mag / (spec_mag + eps)
    cos_sim = F.cosine_similarity(v_specular, v_illumination, dim=1).unsqueeze(1)
    consistency_loss = torch.mean(weight_cos * (1.0 - cos_sim))

    return (lambda_kl * kl_loss + 
            lambda_consistency * consistency_loss + 
            lambda_sparse * sparsity_loss)
