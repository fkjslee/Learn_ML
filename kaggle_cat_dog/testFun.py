import cv2
import torch.nn as nn
import torch

x = torch.rand((20, 2), dtype=torch.float)

y = nn.Softmax(dim=1)(x)
print(torch.topk(x, 1, 1).indices.reshape(-1))
print(torch.topk(y, 1, 1).indices.reshape(-1))
