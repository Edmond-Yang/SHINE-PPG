import torch

def apply_adain_style(content_video, style_mean, style_std):

    b, c, t, h, w = content_video.shape
    content_frames = content_video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

    content_mean = torch.mean(content_frames, dim=[2, 3], keepdim=True)
    content_std = torch.std(content_frames, dim=[2, 3], keepdim=True) + 1e-5

    normalized_content = (content_frames - content_mean) / content_std
    

    style_mean_exp = style_mean.squeeze().unsqueeze(0).expand(b * t, -1).unsqueeze(-1).unsqueeze(-1)
    style_std_exp = style_std.squeeze().unsqueeze(0).expand(b * t, -1).unsqueeze(-1).unsqueeze(-1)

    stylized_frames = normalized_content * style_std_exp + style_mean_exp

    stylized_video = stylized_frames.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return stylized_video