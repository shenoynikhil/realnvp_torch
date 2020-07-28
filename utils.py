import torch
import torch.nn.functional as F
import numpy as np
import torch.distributions as distributions


def logit_transform(x, constraint=0.9, reverse=False):
    '''
    Transforms data from [0, 1] into unbounded space.
    Restricts data into [0.05, 0.95].
    Calculates logit(alpha + (1-alpha) * x).
    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    '''
    if reverse:
        x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
        x -= 0.05    # [0, 0.9]
        x /= 0.9     # [0, 1]
        return x, 0

    B, C, H, W = x.size()
    
    # dequantization
    noise = distributions.Uniform(0., 1.).sample((B, C, H, W))
    x = (x * 255. + noise) / 256.     # [0, 1.0]
    
    # restrict data
    x *= 0.9            # [0, 0.9]
    x += 0.05           # [0.05, 0.95]

    # logit data
    logit_x = torch.log(x) - torch.log(1. - x)

    # log-determinant of Jacobian from the transform
    pre_logit_scale = torch.tensor(np.log(constraint) - np.log(1.0 - constraint))
    log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) - F.softplus(-pre_logit_scale)

    return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))