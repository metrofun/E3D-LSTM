import numpy as np
import torch
import torch.utils.data


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, window=1, horizon=1, transform=None, dtype=torch.float):
        super().__init__()
        self._data = data
        self._window = window
        self._horizon = horizon
        self._dtype = dtype
        self._transform = transform

    def __getitem__(self, index):
        x = self._data[index : index + self._window]
        y = self._data[index + self._window : index + self._window + self._horizon]

        # switching to PyTorch format C,D,H,W
        x = np.swapaxes(x, 0, 1)
        y = np.swapaxes(y, 0, 1)

        if self._transform:
            x = self._transform(x)
            y = self._transform(y)

        return (
            torch.from_numpy(x).type(self._dtype),
            torch.from_numpy(y).type(self._dtype),
        )

    def __len__(self):
        return self._data.shape[0] - self._window - self._horizon + 1
