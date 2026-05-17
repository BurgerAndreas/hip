import torch.nn as nn


class BaseModel(nn.Module):
    def forward(self, data):
        raise NotImplementedError
