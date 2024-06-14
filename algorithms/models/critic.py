from torch import Tensor
import torch.nn as nn


class CriticLayer(nn.Module):
    def __init__(self, input_dim):
        super(CriticLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512, bias=True),
            nn.Tanh(),
            nn.Linear(512, 512, bias=True),
            nn.Tanh(),
            nn.Linear(512, 512, bias=True),
            nn.Tanh(),
            nn.Linear(512, out_features=1, bias=False)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.fc)):
            module = self.fc[i]
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1)
                if i < len(self.fc) - 1:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state_embed: Tensor):
        out = self.fc(state_embed)
        return out


if __name__ == '__main__':
    pass
