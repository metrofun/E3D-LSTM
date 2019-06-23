import h5py
import torch

class SequantialDataset(torch.utils.data.Dataset):

    def __init__(self, data, window=1, horizon=1):
        super().__init__()
        self._data = data
        self._window = window
        self._horizon = horizon

    def __getitem__(self, index):
        x = self._data[index:index + self._window]
        y = self._data[index + self._window:index + self._window + self._horizon]

        return (torch.from_numpy(x), torch.from_numpy(y))

    def __len__(self):
        return self._data.shape[0] - self._window - self._horizon + 1
