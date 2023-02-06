import torch
import torch.nn as nn
from torch.distributions.beta import Beta
from torch.distributions.binomial import Binomial

from parts import distributions


class UnimodalNormal(nn.Module):
    def __init__(self, num_classes, input_dim, dist_func, min_sigma, bins_limit, learn_bin_limit=False):
        super(UnimodalNormal, self).__init__()
        self.num_classes = num_classes
        self.dist_func = dist_func
        self.min_sigma = min_sigma
        self.learn_bin_limit = learn_bin_limit
        if learn_bin_limit:
            self.bins_limit = nn.Parameter(torch.randn(1)).requires_grad_(True)
        else:
            self.bins_limit = bins_limit
        self.mu_output = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Tanh()
        )
        self.sigma = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softplus()
        )
        self.sigma_gp = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softplus()
        )

    def calc_normal_output_probs(self, mu, sig):
        if self.learn_bin_limit:
            bins_limit = torch.sigmoid(self.bins_limit).clamp_min(0.5)
            thresholds = torch.arange(0, self.num_classes + 1, device=mu.device) / self.num_classes * (2 * bins_limit) - bins_limit
        else:
            thresholds = torch.arange(0, self.num_classes + 1, device=mu.device) / self.num_classes * (2 * self.bins_limit) - self.bins_limit
        dist_func_instance = getattr(distributions, self.dist_func)(mu, sig)
        probs = torch.zeros(mu.size(0), self.num_classes, device=mu.device).float()
        for i in range(self.num_classes):
            probs[:, i] = (dist_func_instance.cdf(thresholds[i + 1]) - dist_func_instance.cdf(thresholds[i])).squeeze()
        norm_matrix = torch.diag(1. / torch.sum(probs, dim=1))
        return torch.matmul(norm_matrix, probs)

    def calc_output_probs(self, x):
        mu = self.mu_output(x)
        sig = self.sigma(x).clamp(min=self.min_sigma, max=1e2)
        output_probs = self.calc_normal_output_probs(mu=mu, sig=sig)
        return output_probs

    def forward(self, x):
        return self.calc_output_probs(x)

    def get_outputs(self, x):
        mu = self.mu_output(x)
        sig = self.sigma(x).clamp(min=self.min_sigma, max=1e2)
        sig_gp = self.sigma_gp(x).clamp(min=self.min_sigma, max=1e2)
        output_probs = self.calc_normal_output_probs(mu=mu, sig=sig)
        return output_probs, mu, sig, sig_gp


class UnimodalBinomial(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(UnimodalBinomial, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.tau = 1.0
        self.prob_output = nn.Sequential(
            nn.Linear(input_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        success_values = torch.arange(0, self.num_classes, dtype=x.dtype, device=x.device, requires_grad=False).repeat(x.size(0), 1)
        succes_prob = self.prob_output(x).repeat(1, self.num_classes)
        scores = Binomial(total_count=self.num_classes - 1, probs=succes_prob).log_prob(success_values)
        scores = torch.softmax(scores / self.tau, dim=-1)
        return scores


class UnimodalBeta(nn.Module):
    def __init__(self, num_classes, input_channels):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_output = nn.Sequential(
            nn.Linear(input_channels, 1),
            nn.Softplus()
        )
        self.beta_output = nn.Sequential(
            nn.Linear(input_channels, 1),
            nn.Softplus()
        )
        self.device = None
        self.epsilon = 1e-8

    def calc_unnormalized_beta_cdf(self, b, alpha, beta, npts=100):
        bt = Beta(alpha.float(), beta.float())
        x = torch.linspace(0 + self.epsilon, b - self.epsilon, int(npts * b.cpu().numpy()), device=self.device).float()
        pdf = bt.log_prob(x).exp()
        dx = torch.tensor([1. / (npts * self.num_classes)], device=self.device).float()
        P = pdf.sum(dim=1) * dx
        return P

    def calc_beta_output_probs(self, x):
        thresholds = torch.arange(0, self.num_classes + 1, device=self.device) / self.num_classes * 2 - 1
        alpha = torch.clamp(torch.tensor(1., device=self.device) + self.alpha_output(x), min=1, max=100)
        beta = torch.clamp(torch.tensor(1., device=self.device) + self.beta_output(x), min=1, max=100)
        probs = torch.zeros(alpha.size(0), self.num_classes, device=self.device).float()
        for i in range(0, self.num_classes):
            cdf_next = self.calc_unnormalized_beta_cdf(thresholds[i + 1], alpha, beta)
            cdf_current = self.calc_unnormalized_beta_cdf(thresholds[i], alpha, beta)
            probs[:, i] = cdf_next - cdf_current
        norm_matrix = torch.diag(1. / probs.sum(dim=1))  # normalize probs
        return torch.matmul(norm_matrix, probs)

    def forward(self, x):
        self.device = x.device
        return self.calc_beta_output_probs(x)
