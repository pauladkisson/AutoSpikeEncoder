
from end2end import End2End
from datasets import UnsupervisedDataset
from matplotlib import pyplot as plt
import pickle
import torch

if __name__=='__main__':
    data = UnsupervisedDataset(input_dir='./data/alm1_medium', requested_channels=(4,))

    e2e = End2End(min_k=1, max_k=13, step=1, epochs=50, alpha=1, beta=1, device='cpu', cores=8)
    e2e.fit(data)

    with open('./local/e2e_realdata_ch4.pkl', 'wb') as f:
        pickle.dump(e2e, f)