# Code available at https://github.com/ldeecke/gmm-torch
import torch
import numpy as np
from gmm import GaussianMixture

class autogmm(torch.nn.Module):
    def __init__(self, min_components, max_components, n_features, n_init =10,  use_gpu = False):
        super(autogmm, self).__init__()

        self.min_components = min_components
        self.max_components = max_components
        self.n_features = n_features
        self.n_init = n_init
        self.use_gpu = use_gpu

    def fit(self, x):
        """
        Fits GMM models for number of clusters k = min_components, min_components + 1, ..., max_components.
        Returns model with lowest BIC score.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        return:
            self:       GMM fitted to data with lowest BIC score.
        """
        d = list(x.size())[-1]
        min_bic = np.inf
        for n_components in range(self.min_components, self.max_components + 1):
            for _ in range(self.n_init):
                model = GaussianMixture(n_components, d)
                if self.use_gpu:
                    model.cuda()
                model.fit(x)
                bic = model.bic(x)
                if bic < min_bic:
                    min_bic = bic
                    self.bic_ = bic
                    self.model_ = model
        return self

def likelihood_loss(embeddings, model, gamma = 0.1, theta = 0.05):
    """
    This loss is equal to -log(likelihood of latent representation z given a mixture model fit).
    This is an adaptation of Bo Zong et al. "Deep Autoencoding Gaussian Mixture Model
    For Unsupervised Anomaly Detection" loss function. 
    The idea is that by minimizing the loss we are maximizing the likelihood of observing the latent representation z
    under the provided mixture model.
    args:
            input:      latent representations extracted from encoder,  torch.Tensor (batch_size, d)
            model:      GMM model fit by autogmm(autogmm object) 
            gamma:      scalar multiplier of likelihood loss
            theta:      scalar multiplier for L2 distances between centroids regularization
        return:
            forward pass, log likelohood maximized
    """
    if isinstance(model, autogmm):
        model = model.model_
    if embeddings.is_cuda:
        model.cuda()
    #obtain batch size
    batch_size, _ = embeddings.size()
    _, k, _ = model.mu.size()
    # compute log likelihood
    u = model.mu.permute(1, 0, 2)
    z = embeddings.unsqueeze(0)
    den = torch.rsqrt(torch.abs(2*np.pi*torch.prod(model.var, dim= 2)))
    noexp_num = -0.5*torch.sum((z-u)*(1/model.var).transpose(0,1)*(z-u), dim = 2)
    num = torch.exp(noexp_num)
    likelihood = model.pi*(num.transpose(0,1).unsqueeze(2))*den.unsqueeze(2) # up to here works surely
    likelihood = torch.sum(likelihood.squeeze(2), dim = 1) 
    loglikelihood = - torch.log(likelihood)
    loglikelihood = loglikelihood.view(-1)
    loss1 = torch.sum(loglikelihood) * gamma / batch_size
    print(loss1)
    #Add L2 regularization on distances of centroids as 1/average_distance_centroids
    pairwise_L2_mu = torch.cdist(model.mu, model.mu)
    loss2 = (theta + 1e-12) / (torch.sum(pairwise_L2_mu) / (k**2 - k + 1e-12))
    return loss1 + loss2#consider adding cosine similarity penalty between mean vectors