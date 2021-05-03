import sys
import os
import torch
import torch.nn as nn
import numpy.random as random
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from autoencode import AEEnsemble
from models import Classifier
from datasets import SupervisedDataset, UnsupervisedDataset

cwd = Path(os.path.abspath(__file__))
cwd = cwd.parents[0]
data_dir = os.path.join(cwd, "data")
embeddings_dir = os.path.join(data_dir, "embeddings", "simulated")

sup_data = SupervisedDataset("/export/gaon1/data/jteneggi/DL/pedreira")

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

ae = AEEnsemble(convolutional_encoding=False, device=device, activ=nn.ReLU)
ae.load(prefix="simulated")

for e, d in zip(ae.encoders, ae.decoders):
    e.eval()
    d.eval()

for i, (spikes, targets) in tqdm(enumerate(sup_data)):
    spikes = torch.from_numpy(spikes).float().to(device)
    latent_vecs = [e(spikes) for e in ae.encoders]
    latent = torch.cat(latent_vecs, dim=1).detach().cpu()
    np.save(os.path.join(embeddings_dir, f"session_{i}.npy"), latent)
