import torch
import torch.nn as nn
from .component import *


IN_CH = 3
S_VALUE = 1


class IlluminationModel(nn.Module):
    def __init__(self, name='illumination-branch'):
        super(IlluminationModel, self).__init__()
        self.name = name
        self.encoder_low = LowLevelEncoder()
        self.encoder_high = HighLevelEncoder()
        self.decoder = Decoder()

    def load(self, path_generator):
        self.load_state_dict(torch.load(path_generator(self.name)))

    def save(self, path_generator):
        torch.save(self.state_dict(), path_generator(self.name))

    def forward(self, x):
        feature, parity, skips = self.encoder_low(x)
        feature, _, skips = self.encoder_high(feature, parity, skips)
        video = self.decoder(feature, skips)
        return video



class ReflectanceModel(nn.Module):
    def __init__(self, mode='train'):
        super(ReflectanceModel, self).__init__()
        self.mode = mode
        self.encoder = LowLevelEncoder()
        self.encoder_rppg = HighLevelEncoder()
        self.estimator = Estimator()
        self.encoder_appearance = HighLevelEncoder()
        self.fusion_factor = nn.Parameter(torch.zeros(1))
        self.decoder = Decoder()

    def load(self, path_generator):
        self.encoder.load_state_dict(torch.load(path_generator('reflectance-low-level-encoder')))
        self.encoder_rppg.load_state_dict(torch.load(path_generator('reflectance-rppg-encoder')))
        self.estimator.load_state_dict(torch.load(path_generator('rppg-estimator')))
        
        self.encoder_appearance.load_state_dict(torch.load(path_generator('reflectance-appearance-encoder')))
        self.decoder.load_state_dict(torch.load(path_generator('reflectance-decoder')))
    
        with torch.no_grad():
            self.fusion_factor.copy_(torch.load(path_generator('fusion-factor')))

    def save(self, path_generator):
        torch.save(self.encoder.state_dict(), path_generator('reflectance-low-level-encoder'))
        torch.save(self.encoder_rppg.state_dict(), path_generator('reflectance-rppg-encoder'))
        torch.save(self.estimator.state_dict(), path_generator('rppg-estimator'))
        
        torch.save(self.encoder_appearance.state_dict(), path_generator('reflectance-appearance-encoder'))
        torch.save(self.fusion_factor, path_generator('fusion-factor'))
        torch.save(self.decoder.state_dict(), path_generator('reflectance-decoder'))
        
    def forward(self, x, only_rppg=False):
        feature, parity, skips = self.encoder(x)
        
        feature_rppg, parity_rppg, skips_rppg = self.encoder_rppg(feature, parity, skips)
        signal = self.estimator(feature_rppg, parity_rppg)
        video = None

        if not only_rppg:
            feature_appearance, _, skips_appearance = self.encoder_appearance(feature, parity, skips)
            f_fused = feature_appearance + self.fusion_factor * feature_rppg
            s_fused = [_a + self.fusion_factor * _r for _a, _r in zip(skips_appearance, skips_rppg)]
            video = self.decoder(f_fused, s_fused)

        return video, signal


    # def forward(self, x):
    #     feature, parity, skips = self.encoder(x)
        
    #     feature_rppg, parity_rppg, skips_rppg = self.encoder_rppg(feature, parity, skips)
    #     signal = self.estimator(feature_rppg, parity_rppg)

    #     feature_appearance, _, skips_appearance = self.encoder_appearance(feature, parity, skips)
        
    #     f_fused = feature_appearance + self.fusion_factor * feature_rppg
    #     s_fused = [_a + self.fusion_factor * _r for _a, _r in zip(skips_appearance, skips_rppg)]
        
  
    #     video = self.decoder(f_fused, s_fused)

    #     return video, signal

# class ReflectanceModel(nn.Module):
#     def __init__(self, mode='train'):
#         super(ReflectanceModel, self).__init__()
#         self.mode = mode
#         self.encoder = LowLevelEncoder()
#         self.encoder_rppg = HighLevelEncoder()
#         self.estimator = Estimator()

#         if mode == 'train':
#             self.encoder_appearance = HighLevelEncoder()
#             self.fusion_factor = nn.Parameter(torch.zeros(1))
#             self.decoder = Decoder()

#     def load(self, path_generator):
#         self.encoder.load_state_dict(torch.load(path_generator('reflectance-low-level-encoder')))
#         self.encoder_rppg.load_state_dict(torch.load(path_generator('reflectance-rppg-encoder')))
#         self.estimator.load_state_dict(torch.load(path_generator('rppg-estimator')))
        
#         if self.mode == 'train':
#             self.encoder_appearance.load_state_dict(torch.load(path_generator('reflectance-appearance-encoder')))
#             with torch.no_grad():
#                 self.fusion_factor.copy_(torch.load(path_generator('fusion-factor')))
#             self.decoder.load_state_dict(torch.load(path_generator('reflectance-decoder')))

#     def save(self, path_generator):
#         torch.save(self.encoder.state_dict(), path_generator('reflectance-low-level-encoder'))
#         torch.save(self.encoder_rppg.state_dict(), path_generator('reflectance-rppg-encoder'))

#         if hasattr(self, 'encoder_appearance'):
#             torch.save(self.encoder_appearance.state_dict(), path_generator('reflectance-appearance-encoder'))
#             torch.save(self.fusion_factor, path_generator('fusion-factor'))
#             torch.save(self.decoder.state_dict(), path_generator('reflectance-decoder'))

#         if hasattr(self, 'estimator'):
#             torch.save(self.estimator.state_dict(), path_generator('rppg-estimator'))

#     def forward(self, x):
#         feature, parity, skips = self.encoder(x)
#         feature_rppg, parity_rppg, skips_rppg = self.encoder_rppg(feature, parity, skips)

#         # 計算 rPPG 信號
#         signal = self.estimator(feature_rppg, parity_rppg)

#         # 如果是訓練模式，則計算反射影像輸出
#         video = None
#         if self.mode == 'train':
#             feature_appearance, _, skips_appearance = self.encoder_appearance(feature, parity, skips)
#             # 融合特徵
#             f_fused = feature_appearance + self.fusion_factor * feature_rppg
#             s_fused = [_a + self.fusion_factor * _r for _a, _r in zip(skips_appearance, skips_rppg)]
#             video = self.decoder(f_fused, s_fused)

#         return video, signal