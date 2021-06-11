from torch import nn as nn
import torch

a = torch.LongTensor([1])
# print(a)
embedding = torch.nn.Embedding(2, 5)
# print(embedding.weight)
b = embedding(a)
print(b)
a1 = torch.tensor([[0, 1.]])
a2 = embedding.weight
print(torch.matmul(a1, a2))
