from torch.distributions import *
import torch

def Logistic(a,b):
    base_distribution = Uniform(torch.tensor(0, device=a.device), torch.tensor(1, device=a.device))
    transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
    return TransformedDistribution(base_distribution, transforms)