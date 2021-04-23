import numpy as np

from models import ShallowFFEncoder, ShallowFFDecoder, IntermediateFFEncoder, IntermediateFFDecoder, DeepFFEncoder, DeepFFDecoder
from datasets import UnsupervisedDataset
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class FFAEEnsemble:
    """
    Reproduce the AutoEncoder Paper.
    """

    def __init__(self, optim=None, batch_size=100, epochs=200, lr=.01, device='cpu'):
        self.encoders = [ShallowFFEncoder(), IntermediateFFEncoder(), DeepFFEncoder()]
        self.decoders = [ShallowFFDecoder(), IntermediateFFDecoder(), DeepFFDecoder]
        if not optim:
            optim = torch.optim.SGD
        self.optimizers = [optim(lr=lr,
                                params=list(self.encoders[i].parameters()) +
                                       list(self.decoders[i].parameters()))
                          for i in range(len(self.encoders))]
        self.batch_size = batch_size
        self.training_epochs = epochs
        self.device = device

    def fit(self, x: Dataset, _store_latent=False):
        dataloader = DataLoader(x, batch_size=self.batch_size, shuffle=True)
        loss = torch.nn.MSELoss()
        for epoch in range(self.training_epochs):
            for batch in dataloader.batch_sampler:
                map(lambda o: o.zero_grad(), self.optimizers)
                data = []
                for idx in batch:
                    sample = x[idx]
                    data.append(sample[0])
                spikes = np.concatenate(data, axis=0)
                spikes = torch.from_numpy(spikes)
                latent_vecs = [encoder(spikes) for encoder in self.encoders]
                renconstructed = [decoder(latent_vecs[i]) for i, decoder in enumerate(self.decoders)]
                losses = [loss(spikes, r) for r in renconstructed]
                map(lambda l: l.backward(), losses)
                map(lambda o: o.step(), self.optimizers)

    def predict(self, x: Dataset):
        dataloader = DataLoader(x, batch_size=self.batch_size, shuffle=True)
        for batch in dataloader.batch_sampler:
            data = []
            idxs = []
            for idx in batch:
                sample = x[idx]
                idxs.append(idx)
                data.append(sample[0])
            spikes = np.concatenate(data, axis=0)
            spikes = torch.from_numpy(spikes)
            latent_vecs = [encoder(spikes) for encoder in self.encoders]
            latent = torch.cat(latent_vecs, dim=1)
            x.write(latent, idxs, fname='embeddings_ff_ensemble.npy')
