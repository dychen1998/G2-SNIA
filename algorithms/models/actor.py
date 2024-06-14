from torch import Tensor
import torch.nn as nn


class ActorLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, depth: int = 4):
        super(ActorLayer, self).__init__()
        self.multi_discrete = False
        self.action_outs = []
        modules = [nn.Linear(input_dim, 512, True), nn.Tanh()]

        for i in range(depth):
            modules.append(nn.Linear(512, 512, True))
            modules.append(nn.Tanh())

        modules.append(nn.Linear(512, output_dim, False))
        self.action_outs.append(nn.Sequential(*modules))
        self.action_outs = nn.ModuleList(self.action_outs)
        self.reset_parameters()

    def reset_parameters(self):
        if self.multi_discrete:
            for i in range(len(self.action_outs)):
                fc = self.action_outs[i]
                for j in range(len(fc)):
                    module = fc[j]
                    if isinstance(module, nn.Linear):
                        if j == len(fc) - 1:
                            nn.init.orthogonal_(module.weight, gain=0.01)
                        else:
                            nn.init.orthogonal_(module.weight, gain=1)
                            nn.init.constant_(module.bias, 0)

    def forward(self, feats_input: Tensor) -> Tensor:
        """

        :param feats_input: (B, input_dim) or (input_dim,)
        :return: (B, output_dim) or (output_dim,)
        """
        feats_output = self.action_outs[0](feats_input)

        return feats_output


if __name__ == '__main__':
    pass

