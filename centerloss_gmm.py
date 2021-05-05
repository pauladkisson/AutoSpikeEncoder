#Code taken from https://github.com/KaiyangZhou/pytorch-center-loss and modified for gmm
import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture


class CenterLoss_gmm(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, n_components=10, feat_dim=2, use_gpu=True):
        super(CenterLoss_gmm, self).__init__()
        self.n_components = n_components
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

    def forward(self, x):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
        """
        if self.use_gpu:
            x = x.cuda()

        batch_size = x.size(0)

        n_latent = x.clone().cpu().detach()

        model = GaussianMixture(n_components=self.n_components)
        labels = model.fit_predict(n_latent)
        labels = torch.from_numpy(labels).float().requires_grad_()
        self.centers = torch.from_numpy(model.means_).float().requires_grad_()
        if self.use_gpu:
            labels = labels.cuda()
            self.centers = self.centers.cuda()
        
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.n_components) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.n_components, batch_size).t()
        mat = torch.matmul(x,centers.t())
        distmat = distmat -2*mat

        classes = torch.arange(self.n_components).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.n_components)
        mask = labels.eq(classes.expand(batch_size, self.n_components))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss