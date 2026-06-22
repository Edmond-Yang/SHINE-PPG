import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce

IN_CH = 3
OUT_CH = 3
S_VALUE = 1


def _get_convolution_block(in_channels, hidden_channels, out_channels, pool_size=(2, 2, 2)):
    return nn.Sequential(
        nn.AvgPool3d(kernel_size=pool_size, stride=pool_size, padding=0),
        nn.Conv3d(in_channels, hidden_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
        nn.BatchNorm3d(hidden_channels),
        nn.ELU(),
        nn.Conv3d(hidden_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
        nn.BatchNorm3d(out_channels),
        nn.ELU()
    )


def _get_deconvolution_block(in_channels, hidden_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
        nn.BatchNorm3d(hidden_channels),
        nn.ELU(),
        nn.Conv3d(hidden_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
        nn.BatchNorm3d(out_channels),
        nn.ELU()
    )


def _get_estimator_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1, 1), stride=1,
                  padding=(1, 0, 0)),
        nn.BatchNorm3d(out_channels),
        nn.ELU(),
    )


class LowLevelEncoder(nn.Module):
    def __init__(self, ):
        super(LowLevelEncoder, self).__init__()
        self.in_ch = IN_CH
        stem = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )
        self.encoder_blocks = nn.ModuleList([
            stem,
            # Block 1
            _get_convolution_block(32, 64, 64, (1, 2, 2)),
            # Block 2
            _get_convolution_block(64, 64, 64),
        ])

    def forward(self, x):
        parity = []
        skip_connection = []

        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_connection.append(x)
            if i != 0:  # Skip the stem layer for parity
                parity.append(x.size(2) % 2)

        return x, parity, skip_connection


class HighLevelEncoder(nn.Module):
    def __init__(self, ):
        super(HighLevelEncoder, self).__init__()
        self.in_ch = IN_CH
        self.encoder_blocks = nn.ModuleList([
            # Block 3
            _get_convolution_block(64, 64, 64),
            # Block 4
            _get_convolution_block(64, 64, 64, (1, 2, 2))
        ])

    def forward(self, x, parity, skip_connection):
        skip_connection = skip_connection.copy()
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_connection.append(x)

        return x, parity[::-1], skip_connection[:-1][::-1]


class Decoder(nn.Module):
    def __init__(self, ):
        super(Decoder, self).__init__()

        self.out_ch = OUT_CH
        self.decoder_blocks = nn.ModuleList([
            _get_deconvolution_block(64 + 64, 64, 64),
            _get_deconvolution_block(64 + 64, 64, 64),
            _get_deconvolution_block(64 + 64, 64, 32),
            _get_deconvolution_block(32 + 32, 32, 32)
        ])

        self.final = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=self.out_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, skips):
        for block, skip in zip(self.decoder_blocks, skips):
            target_shape = skip.size()[2:]
            x = F.interpolate(x, size=target_shape, mode='trilinear', align_corners=False)
            x = torch.cat((x, skip), dim=1)
            x = block(x)

        x = self.final(x)
        return x


class Estimator(nn.Module):
    def __init__(self, ):
        super(Estimator, self).__init__()
        self.S = S_VALUE

        self.estimator_blocks = nn.ModuleList([
            _get_estimator_block(64, 64),
            _get_estimator_block(64, 64),
        ])

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, self.S, self.S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

    def forward(self, x, parity):

        for block, t_size in zip(self.estimator_blocks, parity):
            x = F.interpolate(x, scale_factor=(2, 1, 1))
            x = F.pad(x, (0, 0, 0, 0, 0, t_size), mode='replicate')
            x = block(x)

        x = self.final(x)
        x = reduce(x, 'b c t s1 s2 -> b c t', 'mean')[:,-1]
        return x
