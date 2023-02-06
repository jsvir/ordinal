import argparse
import os
import sys
sys.path.append("MedMNIST/")
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

from logger import MetricsLogger
from parts import ptl_modules
from parts.models import catalog
from omegaconf import OmegaConf


class Config:
    def __init__(self, cfg):
        self.__dict__.update(cfg)


def init_class(dataset, method):
    method_class = catalog[method]
    data_module = ptl_modules[dataset]
    return type(method + dataset, (method_class, data_module), {})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


def run_with_config(config, task='train'):
    config = Config(config)
    model_class = init_class(config.dataset, config.method)
    metrics_logger = MetricsLogger(config.output_dir, model_class.__name__)
    for lamba in [0.25, 0.5, 1.0, 0.75]:
        for split in range(config.data_splits):
            for fold in range(config.data_folds):
                config.fold = fold
                config.split = split
                config.eloss_lamda = lamba
                seed_everything(split)  # move to here for random initialization
                if not os.path.exists(config.output_dir): os.makedirs(config.output_dir)
                config.logger = TensorBoardLogger(config.output_dir, name=model_class.__name__, log_graph=False)
                trainer = Trainer.from_argparse_args(config)
                lr_log_callback = LearningRateMonitor(logging_interval='epoch')
                checkpoint_callback = ModelCheckpoint(
                    monitor=config.checkpoint_on,
                    save_top_k=1,
                    mode="min",
                    save_weights_only=True,
                    verbose=True,
                )
                trainer.callbacks.append(lr_log_callback)
                trainer.callbacks.append(checkpoint_callback)
                model = model_class(config)
                if config.error_analysis:
                    model.summary_writer = SummaryWriter(log_dir=f'{config.logger.save_dir}/{config.logger.name}/version_{config.logger.version}')
                if task == 'train':
                    trainer.fit(model)
                trainer.test(model)
                metrics_logger.update(model.test_metrics)
                metrics_logger.write_intermediate(f'{config.logger.save_dir}/{config.logger.name}/version_{config.logger.version}', model.test_metrics)

    metrics_logger.write()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    run_with_config(config)
