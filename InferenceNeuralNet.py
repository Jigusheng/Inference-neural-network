import torch

class Inference(torch.nn.Module):
    def __init__(self, in_dim, interm_dim):
        super().__init__()
        self.W = torch.nn.Parameter(2 * torch.rand(in_dim, interm_dim) - 1, requires_grad=True)
        self.M = torch.nn.Parameter(2 * torch.rand(1, interm_dim) - 1, requires_grad=True)
    def forward(self, x):
        value = x @ self.W
        h = value.shape[1]
        knl_mat = value.view(-1,h,1) @ self.M
        return knl_mat
