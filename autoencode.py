import numpy as np
import os

from models import (
    ShallowFFEncoder,
    ShallowFFDecoder,
    IntermediateFFEncoder,
    IntermediateFFDecoder,
    DeepFFEncoder,
    DeepFFDecoder,
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
from sklearn.mixture import GaussianMixture


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
        activ=torch.nn.Tanh,
    ):
        if convolutional_encoding:
            self.encoders = [
                ShallowConvEncoder(device=device, activ=activ),
                IntermediateConvEncoder(device=device, activ=activ),
                DeepConvEncoder(device=device, activ=activ),
            ]
        else:
            self.encoders = [
                ShallowFFEncoder(device=device, activ=activ),
                IntermediateFFEncoder(device=device, activ=activ),
                DeepFFEncoder(device=device, activ=activ),
            ]
        self.decoders = [
            ShallowFFDecoder(device=device, activ=activ),
            IntermediateFFDecoder(device=device, activ=activ),
            DeepFFDecoder(device=device, activ=activ),
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
            torch.optim.lr_scheduler.StepLR(op, step_size=20, gamma=0.5)
            for op in self.optimizers
        ]
        self.batch_size = batch_size
        self.training_epochs = epochs
        self.device = device

    def fit(self, x: Dataset):
        dataloader = DataLoader(x, batch_size=self.batch_size, shuffle=True)
        loss = torch.nn.MSELoss()
        loss_history = [[] for _ in range(len(self.encoders))]
        for e, d in zip(self.encoders, self.decoders):
            e.train()
            d.train()
        for epoch in range(self.training_epochs):
            epoch_loss = [[] for _ in range(len(self.encoders))]
            print("\nEPOCH " + str(epoch + 1) + " of " + str(self.training_epochs))
            for batch in dataloader.batch_sampler:
                map(lambda o: o.zero_grad(), self.optimizers)
                data = []
                for idx in batch:
                    sample = x[idx]
                    if sample[0].shape[0] > 1:
                        data.append(torch.from_numpy(sample[0]).float())
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
        return loss_history

    def predict(
        self,
        x: Dataset,
        return_embeddings=False,
        save_embeddings=True,
        fname="embeddings_ff_ensemble.npy",
    ):
        dataloader = DataLoader(x, batch_size=1, shuffle=True)
        for e, d in zip(ae.encoders, ae.decoders):
            e.eval()
            d.eval()
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

    def save(self, prefix="", on_drive=False):
        if on_drive:
            base_path = "./gdrive/MyDrive/models"
        else:
            base_path = "models"
        for e in self.encoders:
            torch.save(
                e.state_dict(),
                os.path.join(base_path, f"{prefix}_{e.__class__.__name__}.pth"),
            )
        for d in self.decoders:
            torch.save(
                d.state_dict(),
                os.path.join(base_path, f"{prefix}_{d.__class__.__name__}.pth"),
            )

    def load(self, prefix="", on_drive=False):
        if on_drive:
            base_path = "./gdrive/MyDrive/models"
        else:
            base_path = "models"
        for e in self.encoders:
            e.load_state_dict(
                torch.load(
                    os.path.join(base_path, f"{prefix}_{e.__class__.__name__}.pth")
                )
            )
        for d in self.decoders:
            d.load_state_dict(
                torch.load(
                    os.path.join(base_path, f"{prefix}_{d.__class__.__name__}.pth")
                )
            )
            
    def benchmark(self, min_snr, train_data, test_data, on_drive=False):                
        ###Train AE ensemble, dropping units below min_snr
        dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        loss = torch.nn.MSELoss()
        for e, d in zip(self.encoders, self.decoders):
            e.train()
            d.train()
        for epoch in range(self.training_epochs):
            print("\nEPOCH " + str(epoch + 1) + " of " + str(self.training_epochs))
            for batch in dataloader.batch_sampler:
                map(lambda o: o.zero_grad(), self.optimizers)
                data = []
                for idx in batch:
                    spikes, targets, snrs, num_units = train_data[idx]
                    possible_targets = np.arange(len(snrs))
                    if not(np.any(snrs>=min_snr)): #spike sessions with no units above SNR threshold
                        continue
                    hi_fidel_targets = possible_targets[snrs>=min_snr]
                    spikes = spikes[np.isin(targets, hi_fidel_targets)]
                    if spikes[0].shape[0] > 1:
                        data.append(torch.from_numpy(spikes).float())
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
                    losses[i].backward()
                    self.optimizers[i].step()
            map(lambda s: s.step(), self.schedulers)
        for e, d in zip(self.encoders, self.decoders):
            e.eval()
            d.eval()
        self.save(prefix="benchmark_snr_%s"%min_snr, on_drive=on_drive)
        
        ###Embed test_data using AE ensemble
        latent_vecs = []
        test_targets = []
        for spikes, targets, snrs, num_units in test_data:
            session_latent = []
            possible_targets = np.arange(len(snrs))
            if not(np.any(snrs>=min_snr)): #spike sessions with no units above SNR threshold
                continue
            hi_fidel_targets = possible_targets[snrs>=min_snr]
            spikes = torch.FloatTensor(spikes[np.isin(targets, hi_fidel_targets)])
            session_targets = targets[np.isin(targets, hi_fidel_targets)]
            if "cuda" in self.device:
                spikes = spikes.cuda(0)
            for encoder in self.encoders:
                session_latent.append(encoder(spikes))
            session_latent = torch.cat(session_latent, dim=1).detach().cpu()
            latent_vecs.append(session_latent)
            test_targets.append(session_targets)
        
        return latent_vecs, test_targets
        
        
