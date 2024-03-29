import os
from typing import Tuple, Dict

from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
import torch


class SupervisedDataset(Dataset):
    """
    Load the Pedreira data with ground truth.
    N.B. the original spike traces are downsampled from 79 to the input
    dimension of the autoencoders
    """

    def __init__(self, input_dir: str, input_dim: int = 39):
        self._input_dim = input_dim
        self._input_dir = input_dir
        folders = os.listdir(input_dir)
        self.spike_classes = np.load(
            os.path.join(input_dir, "ground_truth", "spike_classes.npy"),
            allow_pickle=True,
        ).squeeze()
        sessions = list(filter(lambda x: "session" in x, folders))
        self.sessions = sorted(sessions, key=lambda x: int(x.split("_")[1]))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx: int):
        session_dir = os.path.join(self._input_dir, self.sessions[idx])
        channel_dir = os.path.join(session_dir, "channel_0")

        spikes = np.load(
            os.path.join(channel_dir, "spikes.npy"), allow_pickle=True
        ).squeeze()
        spikes = np.array([signal.resample(spike, self._input_dim) for spike in spikes])
        targets = self.spike_classes[idx].squeeze()

        return spikes, targets


class UnsupervisedDataset(Dataset):
    """
    Load the alm1 data, either as single channels or multichannel.
    """

    def __init__(

        self, input_dir: str,
        requested_channels=(0,),
        multichannel: bool = False,
        load_embeddings=None,
        session_idx=[0],
    ):
        sessions = os.listdir(input_dir)
        self._multichannel = multichannel
        # maps a unique index to each example
        self._map = []
        self._input_dir = input_dir
        if load_embeddings:
            self.to_load = load_embeddings
        else:
            self.to_load = "spikes.npy"
        sessions = list(filter(lambda x: "session" in x, sessions))
        sorted_sessions = sorted(sessions, key=lambda x: int(x.split("_")[1]))
        selected_sessions = []
        for idx in session_idx:
            selected_sessions.append(sorted_sessions[idx])
        for session in selected_sessions:
            session_id = int(session.split("_")[1].strip())
            sess_dir = os.path.join(input_dir, session)
            trials = os.listdir(sess_dir)
            trials = list(filter(lambda x: "trial" in x, trials))
            for trial in sorted(trials, key=lambda x: int(x.split("_")[1])):
                trial_id = int(trial.split("_")[1].strip())
                if multichannel:
                    raise NotImplementedError
                else:
                    trial_dir = os.path.join(sess_dir, trial)
                    channels = os.listdir(trial_dir)
                    channels = list(filter(lambda x: "channel" in x, channels))
                    for channel in sorted(channels, key=lambda x: int(x.split("_")[1])):
                        if int(channel.split("_")[1]) in requested_channels:
                            channel_id = int(channel.split("_")[1].strip())
                            self._map.append((session_id, trial_id, channel_id))

    def get_item_info(self, idx: int) -> Dict[str, int]:
        if self._multichannel:
            raise NotImplementedError
        else:
            sess_id, trial_id, channel_id = self._map[idx]
            info = {"session": sess_id, "trial": trial_id, "channel": channel_id}
        return info

    def write(self, idx, embeddings, fname):
        sess_id, trial_id, channel_id = self._map[idx]
        trial_path = os.path.join(
            self._input_dir, "session_" + str(sess_id), "trial_" + str(trial_id)
        )
        channel_path = os.path.join(trial_path, "channel_" + str(channel_id))
        embed = np.array(embeddings)  # should be size num_spikes * embedding_dims
        np.save(os.path.join(channel_path, fname), embed)

    def to_tensor(self):
        data = []
        for i in range(len(self)):
            item, _ = self[i]
            if len(item) > 0:
                data.append(torch.from_numpy(item).float())
        data = torch.cat(data, dim=0)
        return data

    def __len__(self):
        return len(self._map)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple of spikes and spike times
        """
        if self._multichannel:
            raise NotImplementedError
        else:
            sess_id, trial_id, channel_id = self._map[idx]

        trial_path = os.path.join(
            self._input_dir, "session_" + str(sess_id), "trial_" + str(trial_id)
        )
        if self._multichannel:
            raise NotImplementedError
        else:
            channel_path = os.path.join(trial_path, "channel_" + str(channel_id))
            channel_spikes = np.load(os.path.join(channel_path, self.to_load))
            channel_spike_times = np.load(os.path.join(channel_path, "spike_times.npy"))

        return channel_spikes, channel_spike_times


class BenchmarkDataset(Dataset): #add SNR
    """
    Load the Pedreira data with ground truth.
    N.B. the original spike traces are downsampled from 79 to the input
    dimension of the autoencoders.
    Also returns SNR's (snrs) and number of units (num_units)
    """

    def __init__(self, input_dir: str, input_dim: int = 39):
        self._input_dim = input_dim
        self._input_dir = input_dir
        folders = os.listdir(input_dir)
        self.spike_classes = np.load(
            os.path.join(input_dir, "ground_truth", "spike_classes.npy"),
            allow_pickle=True,
        ).squeeze()
        sessions = list(filter(lambda x: "session" in x, folders))
        self.sessions = sorted(sessions, key=lambda x: int(x.split("_")[1]))
        self.num_units = [np.max(sesh_spike_classes) for sesh_spike_classes in self.spike_classes]
        self.num_units = np.array(self.num_units)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx: int):
        session_dir = os.path.join(self._input_dir, self.sessions[idx])
        channel_dir = os.path.join(session_dir, "channel_0")

        spikes = np.load(
            os.path.join(channel_dir, "spikes.npy"), allow_pickle=True
        ).squeeze()
        spikes = np.array([signal.resample(spike, self._input_dim) for spike in spikes])
        snrs = np.load(os.path.join(session_dir, "snrs.npy"), allow_pickle=True)
        targets = self.spike_classes[idx].squeeze()
        num_units = self.num_units[idx]
        return spikes, targets, snrs, num_units
