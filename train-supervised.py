import sys
import os
import torch
import torch.nn as nn
import numpy.random as random
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("./")

from autoencode import AEEnsemble
from models import Classifier
from datasets import SupervisedDataset, UnsupervisedDataset

unsup_data = UnsupervisedDataset("/export/gaon1/data/jteneggi/DL/alm", session_idx=[0])
unsup_spikes = []
for idx, (_unsup_spikes, _) in enumerate(unsup_data):
    if len(_unsup_spikes) > 0:
        # print(_unsup_spikes.shape)
        unsup_spikes.append(_unsup_spikes)
unsup_spikes = np.concatenate(unsup_spikes)
print(unsup_spikes.shape)

sup_spikes, sup_targets = SupervisedDataset("/export/gaon1/data/jteneggi/DL/pedreira")[0]
print(sup_spikes.shape)

spike_batch_size = 128
n_ae_iter = 120
n_class_iter = 6
ratio = n_ae_iter/n_class_iter
sup_L = len(sup_spikes)
unsup_L = len(unsup_spikes)
unsup_epoch_L = int(ratio * sup_L)

print(sup_L, unsup_epoch_L)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

ae = AEEnsemble(batch_size=1, device=device)
# ae.load()

fc = Classifier(num_classes=20, device=device)

num_epochs = 50
lr = 0.0001

torch.autograd.set_detect_anomaly(True)

ae_criterion = nn.MSELoss()
class_criterion = nn.CrossEntropyLoss()

params = list(fc.parameters())
for e, d in zip(ae.encoders, ae.decoders):
    params += list(e.parameters()) + list(d.parameters())
optimizer = torch.optim.Adam(params, lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

class_loss_history = []
ae_losses_history = []

for epoch in range(num_epochs):
    unsup_idx = np.random.choice(range(unsup_L), unsup_epoch_L, replace=False)
    sup_idx = np.random.choice(range(sup_L), sup_L, replace=False)
    # print(unsup_idx, sup_idx)
    print(f"EPOCH {epoch+1} of {num_epochs}")
        
    for e, d in zip(ae.encoders, ae.decoders):
        e.train()
        d.train()
    fc.train()
        
    class_epoch_loss = 0
    ae_epoch_losses = [0 for _ in range(len(ae.encoders))]
        
    unsup_loop_loss = []
    sup_loop_loss = []
    for iter_id in tqdm(range(np.ceil(sup_L / (n_class_iter * spike_batch_size)).astype(int))):
        # unsupervised loop
        loop_loss = []
        for i in range(n_ae_iter):
            optimizer.zero_grad()
            
            spikes = unsup_spikes[unsup_idx[i * spike_batch_size: (i + 1) * spike_batch_size]]
            
            spikes = torch.from_numpy(spikes).float().to(device)
            latent_vecs = [e(spikes) for e in ae.encoders]
            reconstructed = [d(latent_vecs[i]) for i, d in enumerate(ae.decoders)]
            losses = [ae_criterion(spikes, r) for r in reconstructed]
            loss = sum(losses)
            loop_loss.append(loss.detach().cpu().item())
            # print(f"Unsupervised batch {i}, {spikes.shape} spikes")
            loss.backward()
            optimizer.step()
            # print(f"Reconstruction loss: {loss}")
        unsup_loop_loss.append(np.mean(loop_loss))
        # supervised loop
        loop_loss = []
        for i in range(n_class_iter):
            optimizer.zero_grad()
            
            idx = sup_idx[i * spike_batch_size: (i + 1) * spike_batch_size]
            spikes = sup_spikes[idx]
            targets = sup_targets[idx]
            
            spikes = torch.from_numpy(spikes).float().to(device)
            targets = torch.from_numpy(targets).long().to(device)
            
            latent_vecs = [e(spikes) for e in ae.encoders]
            
            latent = torch.cat(latent_vecs, dim=1)
            output = fc(latent)            
            
            reconstructed = [d(latent_vecs[i]) for i, d in enumerate(ae.decoders)]
            ae_losses = [ae_criterion(spikes, r) for r in reconstructed]
            fc_loss = class_criterion(output, targets)
            loss = sum(ae_losses) + fc_loss
            loop_loss.append(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
            # print(f"Reconstruction loss: {sum(ae_losses)}, classification loss: {fc_loss.detach().cpu().item()}")
        sup_loop_loss.append(np.mean(loop_loss))
    print(f"Mean unsupervised classification loss: {np.mean(unsup_loop_loss)}")
    print(f"Mean supervised classification loss: {np.mean(sup_loop_loss)}")
    # fig, axes = plt.subplots(1, 2)
    # ax = axes[0]
    # ax.plot(unsup_loop_loss)
    # ax = axes[1]
    # ax.plot(sup_loop_loss)
    # plt.show()
    scheduler.step()
ae.save()
torch.save(fc.state_dict(), "models/Classifier.pth")
