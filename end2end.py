from typing import Union, List

from models import BaseCoder
import torch
from torch import nn
import random
import numpy as np
from sklearn im

from torch.utils.data import DataLoader, Dataset


def standardize(data: torch.Tensor):
    dm = torch.mean(data, dim=0)
    ds = torch.std(data, dim=0)
    data = (data - dm) / ds
    return data


def SMRE(data: torch.Tensor):
    rt = torch.pow(data, .5)
    rtm = torch.mean(rt, dim=1)
    smre = torch.pow(rtm, 2)
    smre = torch.mean(smre)
    return smre


class End2End(nn.Module):

    def __init__(self, encoders: List[BaseCoder], decoders: List[BaseCoder], cluster_loss_fxn, reconstruction_loss_fxn,
                 min_k=2, max_k=20, epochs=50):
        super().__init__()
        self.ae_optimizer = torch.optim.SGD(lr=1e-5, params=)


class SoftKMeans(nn.Module):
    """
    Via the EM algorithm, finds k clusters
    """

    def __init__(self, k=2, embedding_modules: Union[List[BaseCoder], None] = None, optimize=True,
                 device='cpu', **kwargs):
        super().__init__()
        self.k = k
        self.embedding_modules = embedding_modules
        self.embedding_module = embedding_modules[0]
        self.decoding_module = embedding_modules[1]
        self.device = device
        if torch.cuda.is_available() and "cpu" not in device:
            self.dev = torch.device(device)
        else:
            self.dev = torch.device("cpu")
        if self.embedding_module:
            self.optimizer = torch.optim.SGD(lr=1e-5, params=list(self.embedding_module.parameters()) + list(self.decoding_module.parameters()))
            self.dev = self.embedding_module.dev
        self.epochs = kwargs.get('training_epochs', 1)
        self.batch_size = kwargs.get('batch_size', -1)
        self.centroids = None
        self.optimize = optimize
        self.mse_loss = torch.nn.MSELoss()

    def _initialize_centroids(self, x):
        # randomly select k points as the initial centroids
        dataloader = DataLoader(x, batch_size=self.k, shuffle=True)
        centroids = []
        for sample in dataloader.batch_sampler:
            for i in sample:
                data = x[i]
                if data[0].shape[0] < 1:
                    break
                idx = random.choice(list(range(len(data[0]))))
                example = torch.from_numpy(data[0][idx])[None, :].float()
                if self.embedding_module:
                    with torch.no_grad():
                        example = self.embedding_module(example)
                centroids.append(example)
                if len(centroids) == self.k:
                    self.centroids = torch.cat(centroids, dim=0)
                    return

    def _expectation(self, spikes, optimize_embeddings=True, raw_spikes=None):
        # finds distances from each point to each cluster center, can optimizes encoder to
        # produce embeddings that are more "cluster-like"
        raw_responsibility = torch.cdist(spikes, self.centroids, p=2)
        pairwise = torch.pdist(spikes, p=2)
        if self.embedding_module and optimize_embeddings:
            # We want to produce nice neat clusters while also still maintaining reconstructing
            cluster_score = SMRE(raw_responsibility)
            spread_score = 1 / torch.mean(pairwise.view(-1))
            clustering_objective = cluster_score + spread_score
            if raw_spikes is None:
                clustering_objective.backward(retain_graph=False)
            else:
                reconstructions = self.decoding_module(spikes)
                resonstruction_objective = self.mse_loss(reconstructions, raw_spikes)
                clustering_objective.backward(retain_graph=True)
                resonstruction_objective.backward(retain_graph=False)
            self.optimizer.step()
        return raw_responsibility

    def _maximization(self, spikes, responsibilities):
        # adjust centroids to minimize distances to they're clusters
        classification = torch.argmin(responsibilities, dim=1).reshape(-1)
        centroids = torch.zeros((self.k, spikes[0].shape[0]), device=self.dev, requires_grad=False)
        for i in range(self.k):
            positions = spikes[classification == i]
            with torch.no_grad():
                centroids[i, :] = torch.mean(positions, dim=0)
        # with torch.no_grad():
        #     weights = torch.softmax(1 / responsibilities, dim=1).transpose(0, 1)[:, :, None]
        #     pos_buff = spikes.repeat(self.k, 1, 1)
        #     weighted_positions = torch.multiply(pos_buff, weights)
        #     centroids = torch.sum(weighted_positions, dim=1) / torch.sum(weights, dim=1)
        return centroids

    def fit(self, x):
        if self.batch_size == -1:
            batch_size = len(x)
        else:
            batch_size = self.batch_size
        dataloader = DataLoader(x, batch_size=batch_size, shuffle=True)
        if self.embedding_module:
            if self.optimize:
                self.embedding_module.train()
            else:
                self.embedding_module.eval()
        if self.centroids is None:
            #  centroids must be initialized
            self._initialize_centroids(x)
        for epoch in range(self.epochs):
            centroids = []
            for batch in dataloader.batch_sampler:
                data = []
                for idx in batch:
                    sample = x[idx]
                    if sample[0].shape[0] > 1:
                        data.append(standardize(torch.from_numpy(sample[0]).float()))
                if len(data) == 0:
                    continue
                raw_spikes = torch.cat(data, dim=0)
                if self.embedding_module:
                    self.optimizer.zero_grad()
                    spikes = self.embedding_module(raw_spikes)
                else:
                    spikes = raw_spikes
                responsibility = self._expectation(spikes, raw_spikes=raw_spikes, optimize_embeddings=self.optimize)
                centroids.append(self._maximization(spikes, responsibility)[None, :, :])
            with torch.no_grad():
                self.centroids = torch.mean(torch.cat(centroids, dim=0), dim=0)

    def predict(self, x):
        dataloader = DataLoader(x, batch_size=1, shuffle=True)
        classifications = [list for _ in range(len(dataloader))]
        if self.embedding_module:
            self.embedding_module.eval()
        for batch in dataloader.batch_sampler:
            idx = batch[0]
            idx = int(idx)
            spikes, _ = x[idx]
            if spikes.shape[0] == 0:
                continue
            spikes = torch.from_numpy(spikes).float()
            if self.embedding_module:
                with torch.no_grad():
                    spikes = self.embedding_module(spikes)
            if self.centroids is None:
                raise RuntimeError('Must fit before predicting is possible')
            responsibility = self._expectation(spikes, optimize_embeddings=False)
            classifications[idx] = torch.argmin(responsibility, dim=1).reshape(-1).detach().cpu()
        return classifications
