from typing import Optional

import torch

from algorithms.datasets.base_dataset import BaseDataset


class Cora(BaseDataset):
    def __init__(self, device: Optional[torch.device] = None):
        self.name = 'cora'
        super(Cora, self).__init__(name=self.name, device=device)


if __name__ == '__main__':
    pass
