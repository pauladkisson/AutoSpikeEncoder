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

