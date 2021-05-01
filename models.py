from typing import Union

import torch
from torch import nn
from abc import abstractproperty, abstractmethod


class BaseCoder(nn.Module):
    def __init__(self, input_dim, resize_input, device):
        super().__init__()
        if torch.cuda.is_available() and "cpu" not in device:
            self.dev = torch.device(device)
            print("Using " + device)
        else:
            self.dev = torch.device("cpu")
            print("Using CPU")
        self.input_dim = input_dim
        self.resize_input = resize_input
        self.activ = nn.Tanh()
        self.lin_activ = nn.ReLU


class ShallowFFEncoder(BaseCoder):
    """
    Encode spike shape into low dimensional representation.
    """
    def __init__(self, input_dim=39, resize_input=False, device='cpu'):
        super().__init__(input_dim, resize_input, device)
        self.h16 = nn.Linear(self.input_dim, 16).to(self.dev)
        self.h3 = nn.Linear(16, 3).to(self.dev)

    def forward(self, x):
        x = x.to(self.dev)
        h = self.h16(x)
        h = self.activ(h)
        latent = self.h3(h)
        return latent


class ShallowFFDecoder(BaseCoder):
    """
    Decode latent vector into spike
    """
    def __init__(self, input_dim=39, resize_input=False, device='cpu'):
        super().__init__(input_dim, resize_input, device)
        self.h16 = nn.Linear(3, 16).to(self.dev)
        self.h_out = nn.Linear(16, self.input_dim).to(self.dev)

    def forward(self, x):
        x = x.to(self.dev)
        h = self.h16(x)
        h = self.activ(h)
        out = self.h_out(h)
        return out


class IntermediateFFEncoder(BaseCoder):
    """
    Encode spike shape into low dimensional representation.
    """
    def __init__(self, input_dim=39, resize_input=False, device='cpu'):
        super().__init__(input_dim, resize_input, device)
        self.h16 = nn.Linear(self.input_dim, 16).to(self.dev)
        self.h12 = nn.Linear(16, 12).to(self.dev)
        self.h3 = nn.Linear(12, 3).to(self.dev)

    def forward(self, x):
        x = x.to(self.dev)
        h = self.h16(x)
        h = self.activ(h)
        h = self.h12(h)
        h = self.activ(h)
        latent = self.h3(h)
        return latent


class IntermediateFFDecoder(BaseCoder):
    """
    Decode latent vector into spike
    """
    def __init__(self, input_dim=39, resize_input=False, device='cpu'):
        super().__init__(input_dim, resize_input, device)
        self.h12 = nn.Linear(3, 12).to(self.dev)
        self.h16 = nn.Linear(12, 16).to(self.dev)
        self.h_out = nn.Linear(16, self.input_dim).to(self.dev)

    def forward(self, x):
        x = x.to(self.dev)
        h = self.h12(x)
        h = self.activ(h)
        h = self.h16(h)
        h = self.activ(h)
        out = self.h_out(h)
        return out


class DeepFFEncoder(BaseCoder):
    """
    Encode spike shape into low dimensional representation.
    """
    def __init__(self, input_dim=39, resize_input=False, device='cpu'):
        super().__init__(input_dim, resize_input, device)
        self.h24 = nn.Linear(self.input_dim, 24).to(self.dev)
        self.h16 = nn.Linear(24, 16).to(self.dev)
        self.h12 = nn.Linear(16, 12).to(self.dev)
        self.h3 = nn.Linear(12, 3).to(self.dev)

    def forward(self, x):
        x = x.to(self.dev)
        h = self.h24(x)
        h = self.activ(h)
        h = self.h16(h)
        h = self.activ(h)
        h = self.h12(h)
        h = self.activ(h)
        latent = self.h3(h)
        return latent


class DeepFFDecoder(BaseCoder):
    """
    Decode latent vector into spike
    """
    def __init__(self, input_dim=39, resize_input=False, device='cpu'):
        super().__init__(input_dim, resize_input, device)
        self.h12 = nn.Linear(3, 12).to(self.dev)
        self.h16 = nn.Linear(12, 16).to(self.dev)
        self.h24 = nn.Linear(16, 24).to(self.dev)
        self.h_out = nn.Linear(24, self.input_dim).to(self.dev)

    def forward(self, x):
        x = x.to(self.dev)
        h = self.h12(x)
        h = self.activ(h)
        h = self.h16(h)
        h = self.activ(h)
        h = self.h24(h)
        h = self.activ(h)
        out = self.h_out(h)
        return out


class ShallowConvEncoder(BaseCoder):

    def __init__(self, resize_input=False, device='cpu'):
        super().__init__(None, resize_input, device)
        self.conv3_1_16 = nn.Conv1d(kernel_size=3, padding=1, in_channels=1, out_channels=16).to(self.dev)
        self.mpool3 = nn.MaxPool1d(kernel_size=3, stride=3).to(self.dev)
        self.conv3_16_3 = nn.Conv1d(kernel_size=3, padding=1, in_channels=16, out_channels=3).to(self.dev)
        self.h3 = nn.Linear(16, 3).to(self.dev)

    def forward(self, x):
        x = x.to(self.dev)

        if len(x.shape) == 2:
            x = x.reshape(-1, 1, x.shape[1])

        h = self.conv3_1_16(x)
        h = self.mpool3(h)
        h = self.activ(h)

        h = self.conv3_16_3(h)
        h = self.mpool3(h)
        h = self.activ(h)

        spatial = h.shape[2]

        # reduce to 1 spatial dim
        end_pool = nn.MaxPool1d(kernel_size=spatial)
        latent = end_pool(h).reshape(-1, 3)

        return latent


class IntermediateConvEncoder(BaseCoder):

    def __init__(self, resize_input=False, device='cpu'):
        super().__init__(None, resize_input, device)
        self.conv3_1_16 = nn.Conv1d(kernel_size=3, padding=1, in_channels=1, out_channels=16).to(self.dev)
        self.mpool3 = nn.MaxPool1d(kernel_size=3, stride=3).to(self.dev)
        self.conv3_16_16 = nn.Conv1d(kernel_size=3, padding=1, in_channels=16, out_channels=16).to(self.dev)
        self.conv3_16_3 = nn.Conv1d(kernel_size=3, padding=1, in_channels=16, out_channels=3).to(self.dev)
        self.h3 = nn.Linear(16, 3).to(self.dev)

    def forward(self, x):
        x = x.to(self.dev)

        if len(x.shape) == 2:
            x = x.reshape(-1, 1, x.shape[1])

        h = self.conv3_1_16(x)
        h = self.mpool3(h)
        h = self.activ(h)

        h = self.conv3_16_16(h)
        h = self.mpool3(h)
        h = self.activ(h)

        h = self.conv3_16_3(h)
        h = self.activ(h)
        spatial = h.shape[2]

        # reduce to 1 spatial dim
        end_pool = nn.MaxPool1d(kernel_size=spatial)
        latent = end_pool(h).reshape(-1, 3)

        return latent


class DeepConvEncoder(BaseCoder):

    def __init__(self, resize_input=False, device='cpu'):
        super().__init__(None, resize_input, device)
        self.conv3_1_16 = nn.Conv1d(kernel_size=3, padding=1, in_channels=1, out_channels=16).to(self.dev)
        self.mpool3 = nn.MaxPool1d(kernel_size=3, stride=3).to(self.dev)
        self.mpool2 = nn.MaxPool1d(kernel_size=2, stride=2).to(self.dev)
        self.conv3_16_24 = nn.Conv1d(kernel_size=3, padding=1, in_channels=16, out_channels=16).to(self.dev)
        self.conv3_24_16 = nn.Conv1d(kernel_size=3, padding=1, in_channels=16, out_channels=16).to(self.dev)
        self.conv3_16_3 = nn.Conv1d(kernel_size=3, padding=1, in_channels=16, out_channels=3).to(self.dev)
        self.h3 = nn.Linear(16, 3).to(self.dev)

    def forward(self, x):
        x = x.to(self.dev)

        if len(x.shape) == 2:
            x = x.reshape(-1, 1, x.shape[1])

        h = self.conv3_1_16(x)
        h = self.mpool3(h)
        h = self.activ(h)

        h = self.conv3_16_24(h)
        h = self.mpool3(h)
        h = self.activ(h)

        h = self.conv3_24_16(h)
        h = self.mpool2(h)
        h = self.activ(h)

        h = self.conv3_16_3(h)
        h = self.activ(h)
        spatial = h.shape[2]

        # reduce to 1 spatial dim
        end_pool = nn.MaxPool1d(kernel_size=spatial)
        latent = end_pool(h).reshape(-1, 3)

        return latent
