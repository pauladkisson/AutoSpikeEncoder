from typing import Union, List

from models import BaseCoder
import copy
import sys
from autoencode import AEEnsemble
import torch
from torch import nn
from itertools import chain
import numpy as np
from sklearn.mixture import GaussianMixture
from graspologic.utils import remap_labels
import torch.multiprocessing as mp
from centerloss_gmm import center_loss_fxn


def atom_loss(latent, labels, centroids, repulsion_scale=.0001, attraction_scale=.01):
    point_spread = torch.mean((repulsion_scale / (torch.pdist(latent) + 1e-10) ** 3).view(-1))
    points_from_centroid_attractive = torch.mean((-attraction_scale / (torch.cdist(latent, centroids) + 1e-10) ** 2).view(-1))
    points_from_centroids_repulsive = torch.mean((repulsion_scale / (torch.cdist(latent, centroids) + 1e-10) ** 3).view(-1))
    loss = point_spread + points_from_centroids_repulsive + points_from_centroid_attractive
    return loss


class End2End(nn.Module):
    """
    Class to do end to end training. Makes use of multi-target reconstruction and atom loss
    """
    def __init__(self, cluster_loss_fxn=atom_loss, min_k=2, max_k=20, step=1, alpha=.1, beta=1.25, epochs=100, cores=None, device='cpu'):
        super().__init__()
        if torch.cuda.is_available() and "cpu" not in device:
            self.dev = torch.device(device)
        else:
            self.dev = torch.device("cpu")
        self.AE_initializer = AEEnsemble(convolutional_encoding=True, epochs=200, device=device)
        self.loss_fxn = cluster_loss_fxn
        self.reconstruct_loss = torch.nn.MSELoss()
        self.ae_initialized = False
        self.cluster_fit = False
        self.epochs = epochs
        self.ks = list(range(min_k, max_k, step))
        self.alpha = alpha
        self.beta = beta
        self.num_workers = cores
        self.gmm_models = {k: GaussianMixture(n_components=k) for k in self.ks}
        self.trained_encoders = {}
        self.trained_decoders = {}
        self.reconstruction_losses = {}
        self.center_losses = {}

    def fit_autoencoder(self, x):
        self.AE_initializer.fit(x)
        self.ae_initialized = True

    def fit_k(self, x, k):
        encoders = [copy.deepcopy(e) for e in self.AE_initializer.encoders]
        decoders = [copy.deepcopy(d) for d in self.AE_initializer.decoders]
        gmm = self.gmm_models[k]
        optimizer = torch.optim.SGD(lr=.00001,
                                    params=list(chain.from_iterable([list(encoder.parameters())
                                                                     for encoder in encoders])) +
                                           list(chain.from_iterable([list(decoder.parameters())
                                                                     for decoder in decoders]))
                                    )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=.4)
        raw_spikes = x.to(self.dev)
        recon_loss_history = []
        center_loss_hist = []
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            latent_vecs = [encoder(raw_spikes) for encoder in encoders]
            reconstructions = [decoder(latent_vecs[i]) for i, decoder in enumerate(decoders)]
            reconstruction_loss = torch.zeros(1)
            for i in range(len(encoders)):
                reconstruction_loss = reconstruction_loss + self.reconstruct_loss(raw_spikes, reconstructions[i])
            latent_vecs = torch.cat(latent_vecs, dim=1)
            np_latent_vecs = latent_vecs.detach().clone().numpy()
            labels = torch.from_numpy(gmm.fit_predict(np_latent_vecs))
            centroids = torch.from_numpy(gmm.means_).float()
            cluster_loss = self.loss_fxn(latent_vecs, labels, centroids)
            reconstruction_loss = reconstruction_loss / len(encoders)
            print("Running Epoch #" + str(epoch) + " For K = " + str(k) +
                  "\n Reconstruction Loss: " + str(reconstruction_loss.detach().cpu().item()) +
                  "    Center Loss: " + str(cluster_loss.detach().cpu().item()))
            loss = self.alpha * cluster_loss + self.beta * reconstruction_loss
            recon_loss_history.append(reconstruction_loss.detach().item())
            center_loss_hist.append(cluster_loss.detach().item())
            loss.backward()
            optimizer.step()
            scheduler.step()
        return encoders, decoders, gmm, np.array(recon_loss_history), np.array(center_loss_hist)

    def bics(self, x):
        bics = []
        for k in self.ks:
            _, bic = self.predict(x, k)
            bics.append(bic)
        return np.array(self.ks), np.array(bics)

    def fit(self, X):
        if not self.ae_initialized:
            print("***INITIALIZING AUTOENCODER***")
            self.fit_autoencoder(X)
        if type(X) not in [torch.Tensor, np.ndarray]:
            mem_x = X.to_tensor()
        else:
            mem_x = torch.Tensor(X).float()
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(self.num_workers)
        results = pool.starmap(self.fit_k, list(map(lambda p, y: (p, y), list([mem_x] * len(self.ks)), self.ks)))
        for i, k in enumerate(self.ks):
            encoders, decoders, gmm, rloss_hist, closs_hist = results[i]
            self.trained_encoders[k] = encoders
            self.trained_decoders[k] = decoders
            self.gmm_models[k] = gmm
            self.reconstruction_losses[k] = rloss_hist
            self.center_losses[k] = closs_hist

    def validate(self, cluster_assignments, true_labels):
        predicted_lables = remap_labels(true_labels, cluster_assignments)
        true_positive = np.count_nonzero(predicted_lables == true_labels)
        acc = true_positive / len(true_labels)
        return acc

    def predict(self, x, k, return_latent=False):
        """
        returns bayesian information criterion and cluster assignments for a given k
        Parameters
        ----------
        x

        Returns
        -------

        """
        encoders = self.trained_encoders[k]
        if type(x) not in [torch.Tensor, np.ndarray]:
            data = x.to_tensor()
        else:
            data = torch.Tensor(x).float()
        raw_spikes = data.to(self.dev)
        latent_vecs = [encoder(raw_spikes) for encoder in encoders]
        latent_vecs = torch.cat(latent_vecs, dim=1)
        np_latent_vecs = latent_vecs.detach().clone().numpy()
        assignments = self.gmm_models[k].predict(np_latent_vecs)
        bic = self.gmm_models[k].bic(np_latent_vecs)
        if return_latent:
            return assignments, bic, np_latent_vecs
        else:
            return assignments, bic


if __name__ == '__main__':
    from datasets import UnsupervisedDataset
    from matplotlib import pyplot as plt
    import pickle

    data = UnsupervisedDataset('./data/alm1_medium/', requested_channels=(4,))
    e2e = End2End(min_k=2, max_k=10, epochs=50, device='cpu')
    e2e.fit(data)
    with open('./local/e2e_unsup_mk1.pkl', 'wb') as f:
        pickle.dump(e2e, f)
    ks, bics = e2e.bics(data)
    plt.scatter(ks, bics)
    plt.show()
