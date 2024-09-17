import torch

class MLP(torch.nn.Module):
    def __init__(self, ninputs, nhidden, noutputs):
        super().__init__()
        self.dense1 = torch.nn.Linear(ninputs, nhidden)
        self.dense2 = torch.nn.Linear(nhidden, noutputs)
    def forward(self, x):
        x = self.dense1(x)
        x = torch.nn.functional.relu(x)
        return self.dense2(x)