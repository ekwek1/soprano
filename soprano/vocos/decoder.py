import torch
from torch import nn

from .models import VocosBackbone
from .heads import ISTFTHead


class SopranoDecoder(nn.Module):
    def __init__(self,
                 num_input_channels=512,
                 decoder_num_layers=8,
                 decoder_dim=768,
                 decoder_intermediate_dim=None,
                 hop_length=512,
                 n_fft=2048,
                 upscale=4,
                 dw_kernel=3,
                ):
        super().__init__()
        self.decoder_initial_channels = num_input_channels
        self.num_layers = decoder_num_layers
        self.dim = decoder_dim
        self.intermediate_dim = decoder_intermediate_dim if decoder_intermediate_dim else decoder_dim*3
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.upscale = upscale
        self.dw_kernel = dw_kernel
        
        self.decoder = VocosBackbone(input_channels=self.decoder_initial_channels,
                                      dim=self.dim,
                                      intermediate_dim=self.intermediate_dim,
                                      num_layers=self.num_layers,
                                      input_kernel_size=1,#dw_kernel,
                                      dw_kernel_size=dw_kernel,
                                      )
        self.head = ISTFTHead(dim=self.dim,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length)

    def forward(self, x):
        T = x.size(2)
        x = torch.nn.functional.interpolate(x, size=self.upscale*(T-1)+1, mode='linear', align_corners=True)
        x = self.decoder(x)
        reconstructed = self.head(x)
        return reconstructed
