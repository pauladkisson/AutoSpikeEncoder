import torch

from end2end import End2End
from datasets import SupervisedDataset
from matplotlib import pyplot as plt
import pickle
import torch

if __name__=='__main__':
    data = SupervisedDataset(input_dir='./data/pedreira/', input_dim=39)
    raw_spikes, targets = data[10]
    spikes = torch.from_numpy(raw_spikes).float()
    targets = torch.from_numpy(targets)

    e2e = End2End(min_k=1, max_k=30, step=3, epochs=100, device='cpu', cores=3)
    e2e.fit(raw_spikes)

    with open('./local/e2e_sim_s10_forceloss.pkl', 'wb') as f:
        pickle.dump(e2e, f)
