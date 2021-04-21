import os
from typing import Tuple, Dict

from torch.utils.data import Dataset, DataLoader
import numpy as np


class UnsupervisedDataset(Dataset):
    """
    Load the alm1 data, either as single channels or multichannel.
    """

    def __init__(self, input_dir: str, multichannel: bool = False):
        sessions = os.listdir(input_dir)
        self._multichannel = multichannel
        # maps a unique index to each example
        self._map = []
        self._input_dir = input_dir
        sessions = list(filter(lambda x: "session" in x, sessions))
        for session in sorted(sessions, key=lambda x: int(x.split("_")[1])):
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
                        channel_id = int(channel.split("_")[1].strip())
                        self._map.append((session_id, trial_id, channel_id))

    def get_item_info(self, idx: int) -> Dict[str, int]:
        if self._multichannel:
            raise NotImplementedError
        else:
            sess_id, trial_id, channel_id = self._map[idx]
            info = {"session": sess_id, "trial": trial_id, "channel": channel_id}
        return info

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
            channel_spikes = np.load(os.path.join(channel_path, "spikes.npy"))
            channel_spike_times = np.load(os.path.join(channel_path, "spike_times.npy"))

        return channel_spikes, channel_spike_times
