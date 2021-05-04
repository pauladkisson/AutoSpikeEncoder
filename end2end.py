from typing import Union, List

from models import BaseCoder
import copy
from autoencode import AEEnsemble
import torch
from torch import nn
import random
from itertools import chain
import numpy as np
from sklearn.mixture import GaussianMixture
from multiprocessing import Pool

from torch.utils.data import DataLoader, Dataset


def SMRE(data: torch.Tensor, centroids: torch.Tensor):
    distances = torch.cdist(data, centroids, p=2)
    rt = torch.sqrt(distances)
    rtm = torch.mean(rt, dim=1)
    smre = torch.pow(rtm, 2)
    smre = torch.mean(smre)
    return smre


class End2End(nn.Module):

    def __init__(self, cluster_loss_fxn=SMRE, min_k=2, max_k=20, alpha=.5, beta=.5, epochs=50, batch_size=100,
                 device='cuda:0'):
        super().__init__()
        if torch.cuda.is_available() and "cpu" not in device:
            self.dev = torch.device(device)
        else:
            self.dev = torch.device("cpu")
        self.AE_initializer = AEEnsemble(convolutional_encoding=True, epochs=1, device=device)
        self.loss_fxn = cluster_loss_fxn
        self.reconstruct_loss = torch.nn.MSELoss()
        self.ae_initialized = False
        self.cluster_fit = False
        self.epochs = epochs
        self.batch_size = batch_size
        self.ks = list(range(min_k, max_k))
        self.alpha = alpha
        self.beta = beta
        self.gmm_models = {k: GaussianMixture(n_components=k) for k in self.ks}
        self.trained_encoders = {}
        self.trained_decoders = {}

    def fit_autoencoder(self, x):
        self.AE_initializer.fit(x)
        self.ae_initialized = True

    def fit_k(self, x, k):
        encoders = [copy.deepcopy(e) for e in self.AE_initializer.encoders]
        decoders = [copy.deepcopy(d) for d in self.AE_initializer.decoders]
        gmm = self.gmm_models[k]
        optimizer = torch.optim.SGD(lr=1e-5,
                                    params=list(chain.from_iterable([list(encoder.parameters())
                                                                     for encoder in encoders])) +
                                           list(chain.from_iterable([list(decoder.parameters())
                                                                     for decoder in decoders]))
                                    )

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            data = []
            for sample in x:
                if sample[0].shape[0] > 1:
                    data.append(torch.from_numpy(sample[0]).float())
            if len(data) == 0:
                continue
            raw_spikes = torch.cat(data, dim=0)
            raw_spikes = raw_spikes.to(self.dev)
            latent_vecs = [encoder(raw_spikes) for encoder in encoders]
            reconstructions = [decoder(latent_vecs[i]) for i, decoder in enumerate(decoders)]
            reconstruction_loss = torch.zeros(1)
            for i in range(len(encoders)):
                reconstruction_loss = reconstruction_loss + self.reconstruct_loss(raw_spikes, reconstructions[i])
            latent_vecs = torch.cat(latent_vecs, dim=1)
            gmm.fit(latent_vecs)
            centroids = gmm.means_
            cluster_loss = self.loss_fxn(latent_vecs, centroids)
            reconstruction_loss = reconstruction_loss / len(encoders)
            print("Running Epoch #" + str(epoch) + " For K = " + str(k) +
                  "\n Reconstruction Loss: " + str(reconstruction_loss.detach().cpu().item()) +
                  "    Cluster Loss: " + str(cluster_loss.detach().cpu().item()))
            loss = self.alpha * cluster_loss + self.beta * reconstruction_loss
            loss.backward()
            optimizer.step()
        self.trained_encoders[gmm.n_components] = encoders
        self.trained_decoders[gmm.n_components] = decoders

    def bics(self, x):
        bics = []
        for k in self.ks:
            _, bic = self.predict(x, k)
            bics.append(bic)
        return np.ndarray(self.ks), np.ndarray(bics)

    def fit(self, x):
        if not self.ae_initialized:
            print("***INITIALIZING AUTOENCODER***")
            #self.fit_autoencoder(x)
        pool = Pool()
        pool.starmap(self.fit_k, list(map(lambda p, y: (p, y), list([Dataset]*len(self.ks)), self.ks)))

    def predict(self, x, k):
        """
        returns bayesian information criterion and cluster assignments for a given k
        Parameters
        ----------
        x

        Returns
        -------

        """
        data = []
        try:
            encoders = self.trained_encoders[k]
        except IndexError:
            raise RuntimeError("Must fit before predicting.")
        for sample in x:
            if sample[0].shape[0] > 1:
                data.append(torch.from_numpy(sample[0]).float())
        if len(data) == 0:
            raise ValueError
        raw_spikes = torch.cat(data, dim=0)
        raw_spikes = raw_spikes.to(self.dev)
        latent_vecs = [encoder(raw_spikes) for encoder in encoders]
        assignments = self.gmm_models[k].predict(latent_vecs)
        bic = self.gmm_models[k].bic(latent_vecs)
        return assignments, bic


if __name__ == '__main__':
    from datasets import UnsupervisedDataset
    from matplotlib import pyplot as plt
    import pickle
    data = UnsupervisedDataset('./data/alm1/')
    e2e = End2End(min_k=2, max_k=3, epochs=1, device='cpu')
    e2e.fit(data)
    ks, bics = e2e.bics(data)
    plt.plot(ks, bics)
    with open('./local/e2e_unsup_mk1.pkl', 'wb') as f:
        pickle.dump(e2e, f)



