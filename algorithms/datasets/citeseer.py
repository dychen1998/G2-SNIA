from typing import Optional

import torch

from algorithms.datasets.base_dataset import BaseDataset


class Citeseer(BaseDataset):
    def __init__(self, device: Optional[torch.device] = None):
        self.name = 'citeseer'
        split_seed = 2
        super(Citeseer, self).__init__(name=self.name, split_seed=split_seed, device=device)


if __name__ == '__main__':
    pass
