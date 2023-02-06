import numpy as np
import os


class MetricsLogger(object):
    def __init__(self, out_dir, model_type):
        self.metrics = {
            'test_mae': [],
            'test_accuracy': [],
            'test_oneoff_accuracy': [],
            'test_unimodality': [],
            'test_entropy_ratio': [],
            'test_kappa': [],
            'test_spearman': [],
        }
        self.out_file = os.path.join(out_dir, model_type, 'test_results.txt')

    def update(self, result: dict):
        for k, v in result.items():
            self.metrics[k].append(v.item())

    def write_intermediate(self, workdir, result):
        with open(f'{workdir}/RESULTS.txt', mode='w') as w:
            for k, v in result.items():
                w.write(f'{k} {v:.4f}\n')

    def write(self):
        with open(self.out_file, mode='a') as w:
            for k, v in self.metrics.items():
                w.write('{}: {}\n'.format(k, ['{:.4f}'.format(val) for val in v]))
                w.write('{}: mean = {:.4f}, std = {:.4f}\n'.format(k, np.mean(v).item(), np.std(v).item()))
