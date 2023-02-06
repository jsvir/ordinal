from torchvision.models import resnet101, resnet18
from parts.dataset import dataset
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics import CohenKappa, SpearmanCorrCoef
from parts.metrics import ExactAccuracy, MAE, OneOffAccuracy, EntropyRatio, Unimodality


class BaseModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.transforms = None
        self.backbone_model = None
        self.transition_layer = None
        self.weights = None
        self.output_layers = None
        self.loss_func = None
        self.data_splits = None

        for k, v in vars(config).items():
            if not k.startswith('_'): self.hparams.__setattr__(k, v)

        self.val_entropy_ratio = EntropyRatio(output_logits=config.output_logits)

        self.test_mae = MAE()
        self.test_accuracy = ExactAccuracy()
        self.test_one_off_accuracy = OneOffAccuracy()
        self.test_entropy_ratio = EntropyRatio(output_logits=config.output_logits)
        self.test_unimodality = Unimodality(output_logits=config.output_logits)

        self.test_kappa = CohenKappa(num_classes=config.num_classes, weights='quadratic')
        self.test_spearman = SpearmanCorrCoef()

        self.test_metrics = {}
        self.summary_writer = None

    def forward(self, x):
        x = self.backbone_model(x)
        x = self.transition_layer(x)
        x = self.output_layers(x)
        return x

    def build_data_loader(self, task):
        if task == 'train':
            shuffle = True
            batch_size = self.config.train_batch_size
            workers = self.config.train_workers
        elif task == 'test':
            shuffle = False
            batch_size = self.config.test_batch_size
            workers = self.config.test_workers
        elif task == 'val':
            shuffle = False
            batch_size = self.config.val_batch_size
            workers = self.config.val_workers
        transformed_dataset = self.dataset_class(self.config, transforms.Compose(self.transforms_list[task]), task,
                                                 self.data_splits)
        sampler = None
        if task == 'train':
            if not self.config.not_weigted:
                labels = transformed_dataset.get_labels()
                class_props = np.array([np.sum(labels == i) for i in range(self.config.num_classes)]) / len(labels)
                class_weights = 1 / class_props
                class_weights /= np.sum(class_weights)
                class_weights = torch.tensor(class_weights).float()
            else:
                class_weights = torch.ones(self.config.num_classes).float()
            return DataLoader(transformed_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                              drop_last=True), class_weights

        return DataLoader(transformed_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                          num_workers=workers, drop_last=True)

    def train_dataloader(self):
        loader, self.weights = self.build_data_loader('train')
        return loader

    def val_dataloader(self):
        return self.build_data_loader('val')

    def test_dataloader(self):
        return self.build_data_loader('test')

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].float()
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def log_image_to_tb(self, images, true_labels, pred_labels):
        for i, image in enumerate(images):
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            self.summary_writer.add_image(f'pred: {pred_label} label: {true_label}', image, self.current_epoch)
        self.summary_writer.flush()

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].float()
        y_hat = self(x)
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]
        self.val_entropy_ratio(y_hat, y)
        self.log_dict({
            'val_loss': self.loss_func(y_hat, y),
            'val_mae': MAE()(y_hat, y),
            'val_accuracy': ExactAccuracy()(y_hat, y),
            'val_oneoff_accuracy': OneOffAccuracy()(y_hat, y),
            'val_unimodality': Unimodality(output_logits=self.config.output_logits)(y_hat),
            'val_kappa': CohenKappa(num_classes=self.config.num_classes, weights='quadratic').to(y_hat.device)(y_hat,
                                                                                                               y.int()),
            'val_spearman': SpearmanCorrCoef().to(y_hat.device)(y_hat.argmax(dim=1).float(), y.float()),

        }, on_step=False, on_epoch=True)

    def validation_epoch_end(self, results):
        self.log_dict({
            'val_entropy_ratio': self.val_entropy_ratio.compute(),
        }, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].float()
        y_hat = self(x)
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]
        if self.hparams.error_analysis:
            preds_classes = torch.argmax(y_hat, dim=-1)
            wrong_preds = x[preds_classes != y]
            true_labels = y[preds_classes != y]
            pred_labels = preds_classes[preds_classes != y]
            self.log_image_to_tb(wrong_preds, true_labels, pred_labels)

        self.test_entropy_ratio(y_hat, y)
        self.test_mae(y_hat, y)
        self.test_accuracy(y_hat, y)
        self.test_one_off_accuracy(y_hat, y)
        self.test_unimodality(y_hat)
        self.test_kappa.to(y_hat.device)(y_hat, y.int())
        self.test_spearman.to(y_hat.device)(y_hat.argmax(dim=1).float(), y.float())

    def test_epoch_end(self, results):
        self.test_metrics = {
            'test_mae': self.test_mae.compute(),
            'test_accuracy': self.test_accuracy.compute(),
            'test_oneoff_accuracy': self.test_one_off_accuracy.compute(),
            'test_unimodality': self.test_unimodality.compute(),
            'test_entropy_ratio': self.test_entropy_ratio.compute(),
            'test_kappa': self.test_kappa.compute(),
            'test_spearman': self.test_spearman.compute(),
        }
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def on_train_end(self):
        if self.hparams.error_analysis:
            self.summary_writer.close()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=self.config.gamma,
                                                              milestones=[int(v) for v in
                                                                          self.config.lr_sched.split(',')]),
            'interval': 'epoch',
            'frequency': 1}
        return [optimizer], [scheduler]


class HCIModule(BaseModule):
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config
        self.backbone_model = resnet18(pretrained=True)
        self.transforms_list = {
            'train': [
                transforms.RandomAffine(degrees=180., translate=(0.2, 0.2), scale=(0.75, 1.2)),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
            'val': [
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
            'test': [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        }
        self.dataset_class = dataset.HCIDataset
        self.save_hyperparameters()
        self.data_splits = dataset.HCIDataset.split_dataset(self.config.data_images, self.config.split)


class AdienceModule(BaseModule):
    def __init__(self, config=None):
        super().__init__(config)
        self.backbone_model = resnet101(pretrained=True)
        self.config = config
        self.transforms_list = {
            'train': [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()],
            'val': [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor()],
            'test': [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()]
        }
        self.dataset_class = dataset.AdienceDataset
        self.save_hyperparameters()


class FGNETModule(BaseModule):
    def __init__(self, config=None):
        super().__init__(config)
        self.backbone_model = resnet18(pretrained=True)
        self.config = config
        self.transforms_list = {
            'train': [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()],
            'val': [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor()],
            'test': [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()]
        }
        self.dataset_class = dataset.FGNETDataset
        self.save_hyperparameters()
        self.data_splits = dataset.FGNETDataset.split_dataset(self.config.root_dir, self.config.split)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, input_dim=3 * 784, out_dim=128, hidden_dim=32):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x.reshape(-1, 3 * 28 * 28))


class MedMNISTModule(BaseModule):
    def __init__(self, config=None):
        super().__init__(config)
        self.backbone_model = resnet18(pretrained=True)
        self.config = config
        self.transforms_list = {
            'train': [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=180, translate=(0.05, 0.05), scale=(0.98, 1.02),
                                        fillcolor=(128, 128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ],
            'val': [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ],
            'test': [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        }
        self.dataset_class = dataset.MedMNISTOrdinal
        self.save_hyperparameters()
        self.data_splits = {
            'train': getattr(dataset, config.medmnist_type)(
                split='train',
                transform=transforms.Compose(self.transforms_list['train']),
                download=True,
                as_rgb=True,
                root=config.root_dir),

            'val': getattr(dataset, config.medmnist_type)(
                split='val',
                transform=transforms.Compose(self.transforms_list['val']),
                download=True,
                as_rgb=True,
                root=config.root_dir),
            'test': getattr(dataset, config.medmnist_type)(
                split='test',
                transform=transforms.Compose(self.transforms_list['test']),
                download=True,
                as_rgb=True,
                root=config.root_dir)
        }


ptl_modules = {
    'HCI': HCIModule,
    'Adience': AdienceModule,
    'FGNET': FGNETModule,
    'MedMNISTOrdinal': MedMNISTModule
}
