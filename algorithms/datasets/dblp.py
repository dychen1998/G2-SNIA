from typing import Optional

import torch

from algorithms.datasets.base_dataset import BaseDataset


class DBLP(BaseDataset):
    def __init__(self, device: Optional[torch.device] = None):
        self.name = 'dblp'
        super(DBLP, self).__init__(name=self.name, device=device)


if __name__ == '__main__':
    pass
