import numpy as np
import os

from models import (
    ShallowFFEncoder,
    ShallowFFDecoder,
    IntermediateFFEncoder,
    IntermediateFFDecoder,
    DeepFFEncoder,
    DeepFFDecoder,
    ShallowConvEncoder,
    IntermediateConvEncoder,
    DeepConvEncoder,
)
import torch
from torch.utils.data import DataLoader, Dataset


def standardize(data: torch.Tensor):
    dm = torch.mean(data, dim=0)
    ds = torch.std(data, dim=0)
    data = (data - dm) / ds
    return data


class AEEnsemble:
    """
    Reproduce the AutoEncoder Paper.
    """

    def __init__(
        self,
        optim=None,
        convolutional_encoding=True,
        batch_size=100,
        epochs=200,
        lr=(0.01, 0.01, 0.01),
        device="cpu",
    ):
        if convolutional_encoding:
            self.encoders = [
                ShallowConvEncoder(device=device),
                IntermediateConvEncoder(device=device),
                DeepConvEncoder(device=device),
            ]
        else:
            self.encoders = [
                ShallowFFEncoder(device=device),
                IntermediateFFEncoder(device=device),
                DeepFFEncoder(device=device),
            ]
        self.decoders = [
            ShallowFFDecoder(device=device),
            IntermediateFFDecoder(device=device),
            DeepFFDecoder(device=device),
        ]
        if not optim:
            optim = torch.optim.SGD
        self.optimizers = [
            optim(
                lr=lr[i],
                params=list(self.encoders[i].parameters())
                + list(self.decoders[i].parameters()),
            )
            for i in range(len(self.encoders))
        ]
        self.schedulers = [
            torch.optim.lr_scheduler.StepLR(op, step_size=10, gamma=0.1)
            for op in self.optimizers
        ]
        self.batch_size = batch_size
        self.training_epochs = epochs
        self.device = device

    def fit(self, x: Dataset):
        dataloader = DataLoader(x, batch_size=self.batch_size, shuffle=True)
        loss = torch.nn.MSELoss()
        loss_history = [[] for _ in range(len(self.encoders))]
        map(lambda e: e.train(), self.encoders)
        map(lambda d: d.train(), self.decoders)
        for epoch in range(self.training_epochs):
            epoch_loss = [[] for _ in range(len(self.encoders))]
            print("\nEPOCH " + str(epoch + 1) + " of " + str(self.training_epochs))
            for batch in dataloader.batch_sampler:
                map(lambda o: o.zero_grad(), self.optimizers)
                data = []
                for idx in batch:
                    sample = x[idx]
                    if sample[0].shape[0] > 1:
                        data.append(standardize(torch.from_numpy(sample[0]).float()))
                if len(data) == 0:
                    continue
                spikes = torch.cat(data, dim=0)
                if "cuda" in self.device:
                    spikes = spikes.cuda(0)
                latent_vecs = [encoder(spikes) for encoder in self.encoders]
                renconstructed = [
                    decoder(latent_vecs[i]) for i, decoder in enumerate(self.decoders)
                ]
                losses = [loss(spikes, r) for r in renconstructed]
                for i in range(len(losses)):
                    epoch_loss[i] = losses[i].detach().cpu().item()
                    losses[i].backward()
                    self.optimizers[i].step()
            for i, ae_loss in enumerate(epoch_loss):
                loss_history[i].append(np.mean(ae_loss))
            map(lambda s: s.step(), self.schedulers)
        loss_history = [np.array(ae_hist) for ae_hist in loss_history]
        map(lambda o: o.zero_grad(), self.optimizers)
        return loss_history

    def predict(
        self,
        x: Dataset,
        return_embeddings=False,
        save_embeddings=True,
        fname="embeddings_ff_ensemble.npy",
    ):
        dataloader = DataLoader(x, batch_size=1, shuffle=True)
        map(lambda e: e.eval(), self.encoders)
        map(lambda d: d.eval(), self.decoders)
        embeddings = [None] * len(x)
        for batch in dataloader.batch_sampler:
            idx = batch[0]
            idx = int(idx)
            spikes, _ = x[idx]
            if spikes.shape[0] == 0:
                continue
            spikes = torch.from_numpy(spikes).float()
            spikes = standardize(spikes)
            if "cuda" in self.device:
                spikes = spikes.cuda(0)
            latent_vecs = [encoder(spikes) for encoder in self.encoders]
            latent = torch.cat(latent_vecs, dim=1).detach().cpu()
            if save_embeddings:
                x.write(idx, latent, fname=fname)
            if return_embeddings:
                embeddings[idx] = latent
        if return_embeddings:
            return embeddings
        else:
            return None

    def save(self):
        for e in self.encoders:
            torch.save(
                e.state_dict(), os.path.join("models", f"{e.__class__.__name__}.pth")
            )
        for d in self.decoders:
            torch.save(
                d.state_dict(), os.path.join("models", f"{d.__class__.__name__}.pth")
            )

    def load(self):
        for e in self.encoders:
            e.load_state_dict(
                torch.load(os.path.join("models", f"{e.__class__.__name__}.pth"))
            )
        for d in self.decoders:
            d.load_state_dict(
                torch.load(os.path.join("models", f"{d.__class__.__name__}.pth"))
            )
