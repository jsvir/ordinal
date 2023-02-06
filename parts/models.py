import torch
import torch.nn as nn
import numpy as np

from parts import BaseModule, UnimodalUniformOTLoss, SORDLoss, DLDLLoss, UnimodalNormal, OTLoss, OTLossSoft, UnimodalBeta, UnimodalBinomial, EntropyLoss


class UNIORD(BaseModule):
    """
     Optimal Transport Loss and Unimodal OutputProbabilities Model
    """

    def __init__(self, config):
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = UnimodalNormal(config.num_classes, config.ordinal_input_dim, config.dist_func, config.min_sigma, config.bins_limit)
        self.loss_func = OTLoss(self.hparams.num_classes)

class UNIORDEnt(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = UnimodalNormal(config.num_classes, config.ordinal_input_dim, config.dist_func, config.min_sigma, config.bins_limit)
        self.loss_func = OTLoss(self.hparams.num_classes)
        self.entropy_loss = EntropyLoss()

    def forward(self, x):
        latent_x = self.backbone_model(x)
        x = self.transition_layer(latent_x)
        y_probs, mu, sigma, sigma_gp = self.output_layers.get_outputs(x)
        return y_probs, mu, sigma, sigma_gp, latent_x

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].float()
        y_probs, mu, sigma, sigma_gp, latent_x = self(x)
        ot_loss = self.loss_func(y_probs, y)
        eloss = self.entropy_loss(y_probs, y)
        loss = ot_loss + self.hparams.eloss_lamda * eloss
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('ot_loss', ot_loss.item(), on_epoch=True)
        self.log('entropy_loss', eloss.item(), on_epoch=True)

        # log some stats on train set
        thresholds = torch.arange(0, self.hparams.num_classes + 1, device=y.device) / self.hparams.num_classes * (2 * self.hparams.bins_limit) - self.hparams.bins_limit
        mu_dist_correct, mu_dist_incorrect = [], []
        mu_correct = mu[y_probs.argmax(-1) == y]
        mu_incorrect = mu[y_probs.argmax(-1) != y]
        y_probs_correct = y_probs[y_probs.argmax(-1) == y]
        y_probs_incorrect = y_probs[y_probs.argmax(-1) != y]

        for i, m in enumerate(mu_correct):
            mu_dist_correct.append(min(m - thresholds[y_probs_correct[i].argmax(-1)], thresholds[y_probs_correct[i].argmax(-1)+1] - m).item())

        for i, m in enumerate(mu_incorrect):
            mu_dist_incorrect.append(min(m - thresholds[y_probs_incorrect[i].argmax(-1)], thresholds[y_probs_incorrect[i].argmax(-1)+1] - m).item())
        if len(mu_dist_correct) != 0 and len(mu_dist_incorrect) != 0:
            self.log('mu_dist_correct_div_incorrect', np.mean(mu_dist_correct) / np.mean(mu_dist_incorrect), on_epoch=True)

        sigma_correct = sigma[y_probs.argmax(-1) == y]
        sigma_incorrect = sigma[y_probs.argmax(-1) != y]
        if len(sigma_correct) != 0 and len(sigma_incorrect) != 0:
            self.log('sigma_ratio_correct_div_incorrect', sigma_correct.mean().item() / sigma_incorrect.mean().item(), on_epoch=True)

        self.log('mu', mu.mean().item(), on_epoch=True, on_step=False)
        self.log('sigma', sigma.mean().item(), on_epoch=True, on_step=False)
        return loss

class UNIORDBinomial(BaseModule):
    """
     Optimal Transport Loss and Unimodal OutputProbabilities Model
    """

    def __init__(self, config, ordinal_input_dim=1000):
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = UnimodalBinomial(self.hparams.num_classes, ordinal_input_dim)
        self.loss_func = OTLoss(self.hparams.num_classes)


class UNIORDSoft(UNIORD):
    def __init__(self, config, ordinal_input_dim=1000):
        super().__init__(config, ordinal_input_dim)
        self.loss_func = OTLossSoft(self.hparams.num_classes)


class UNIORDBetaSoft(UNIORD):
    def __init__(self, config):
        super().__init__(config)
        self.output_layers = UnimodalBeta(self.hparams.num_classes, 1000)
        self.loss_func = OTLossSoft(self.hparams.num_classes)


class SORD(BaseModule):
    """
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf
    """

    def __init__(self, config):
        config.output_logits = True
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = nn.Linear(1000, config.num_classes)
        self.loss_func = SORDLoss(config.num_classes)


class Liu(BaseModule):
    """
    https://arxiv.org/pdf/1911.02475.pdf
    """

    def __init__(self, config):
        config.output_logits = True
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = nn.Linear(1000, config.num_classes)
        self.loss_func = UnimodalUniformOTLoss(config.num_classes)


class DLDL(BaseModule):
    """
    https://arxiv.org/pdf/1611.01731.pdf
    """

    def __init__(self, config):
        config.output_logits = True
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = nn.Linear(1000, config.num_classes)
        self.loss_func = DLDLLoss(config.num_classes)


class BeckhamBinomial(BaseModule):
    """
    http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf
    According to the paper Binomial dist produces the best results
    """

    def __init__(self, config, ordinal_input_dim=1000):
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = UnimodalBinomial(self.hparams.num_classes, ordinal_input_dim)
        self.loss_func = OTLoss(self.hparams.num_classes)


models_list = [
    Liu, BeckhamBinomial, UNIORD, SORD, DLDL, UNIORDBinomial, UNIORDSoft, UNIORDEnt
]

catalog = {m.__name__: m for m in models_list}
