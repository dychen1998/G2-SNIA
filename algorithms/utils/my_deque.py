from typing import Optional

import torch
from torch import Tensor


class MyDeque:
    def __init__(self, num_envs: int, max_len: int, device: Optional[torch.device] = None):
        self.num_envs = num_envs
        self.max_len = max_len
        self.device = device
        self.deque = torch.zeros((num_envs, max_len), dtype=torch.long, device=self.device) - 1
        self.env_index = torch.arange(num_envs, dtype=torch.long, device=self.device).unsqueeze(dim=1)
        self.env_index = self.env_index.repeat(1, max_len)

    def append(self, pos: int,  x: Tensor) -> None:
        """
        添加元素

        :param pos:  添加元素的位置
        :param x:  (num_envs, )  添加的内容
        :return None
        """
        self.deque[:, pos] = x

    def clear(self):
        """
        清空队列

        """
        self.deque[:] = -1

    def __len__(self):
        return self.deque.shape[0]

    def __ge__(self, other):
        return self.deque > other

    def __getitem__(self, i):
        return self.deque[i], self.env_index[i]











