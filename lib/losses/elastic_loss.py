import torch
from torch import nn
import torch.nn.functional as F
"""
Adapted from Nerfies by Yushi
"""
# @staticmethod
def calculate_deformation_gradient(points, offset):
    # import pdb; pdb.set_trace()
    u = offset[..., 0]
    v = offset[..., 1]
    w = offset[..., 2]

    grad_outputs = torch.ones_like(u)
    grad_u = torch.autograd.grad(u, [points],
                                 grad_outputs=torch.ones_like(grad_outputs),
                                 create_graph=True)[0]
    grad_v = torch.autograd.grad(v, [points],
                                 grad_outputs=torch.ones_like(grad_outputs),
                                 create_graph=True)[0]
    grad_w = torch.autograd.grad(w, [points],
                                 grad_outputs=torch.ones_like(grad_outputs),
                                 create_graph=True)[0]

    grad_deform = torch.stack([grad_u, grad_v, grad_w], -1)  #

    return grad_deform

# loss 部分
class JacobianSmoothness(nn.Module):
    # Directly Panalize the grad of D_field Jacobian.
    def __init__(self, margin=0.5):
        super().__init__()
        # self.gradient_panelty = torch.nn.MSELoss()
        self.margin = margin

    def forward(self, gradient: torch.Tensor):
        """eikonal loss to encourage smoothness
        Args:
            gradient (torch.Tensor): B N 3?
        Returns:
            torch.Tensor: max(||gradient.norm()||_2^{2}-margin, 0)
        """
        # ?
        # import ipdb
        # ipdb.set_trace()
        # return self.gradient_panelty(torch.linalg.norm(gradient, dim=1), 1)
        grad_norm = torch.linalg.norm(gradient, dim=1).square()  # B 3 3
        return F.relu(grad_norm - self.margin).mean()