import glob
import math
import os
import random
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from medmnist.dataset import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    import cv2
except:
    pass


class OrdinalDataset(Dataset, ABC):
    def __init__(self, config, transform, task, data_splits):
        self.config = config
        self.transform = transform
        self.task = task
        self.data_splits = data_splits

    @abstractmethod
    def get_labels(self):
        pass


class FGNETDataset(OrdinalDataset):
    def __init__(self, config, transform, task, data_splits):
        super().__init__(config, transform, task, data_splits)
        self.X, self.Y = data_splits[task]

    def get_labels(self):
        return np.array([self.Y])

    def __len__(self):
        return len(self.X)

    @staticmethod
    def split_dataset(root_dir, seed=0):
        imgs = glob.glob(f"{root_dir}/images/*.JPG")
        X = imgs
        Y = []
        for f in imgs:
            age = os.path.basename(f).split('A')[1].split('.')[0]
            try:
                age = int(age)
            except:
                age = int(''.join(age[:-1]))
            Y.append(math.floor(age / 10))
        print(np.max(Y))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=100, random_state=seed)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=100, random_state=seed)
        print('size(X_train)={}'.format(len(X_train)))
        print('size(X_test)={}'.format(len(X_test)))
        print('size(X_val)={}'.format(len(X_val)))
        return {
            'train': (X_train, Y_train),
            'test': (X_test, Y_test),
            'val': (X_val, Y_val),
        }

    def __getitem__(self, idx):
        imgpath = self.X[idx]
        label = self.Y[idx]
        image = Image.open(imgpath)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label
        }


class AdienceDataset(OrdinalDataset):
    def __init__(self, config, transform, task, data_splits):
        super().__init__(config, transform, task, data_splits)
        self.root_dir = config.root_dir
        self.images_dir = config.data_images
        self.fold = config.fold
        self.XY = self.read_from_txt_file()

    def get_labels(self):
        return np.array([t[1] for t in self.XY])

    def __len__(self):
        return len(self.XY)

    def read_from_txt_file(self):
        txt_file = f'{self.root_dir}/train_val_txt_files_per_fold/test_fold_is_{self.fold}/age_{self.task}.txt'
        data = []
        f = open(txt_file)
        for line in f.readlines():
            image_file, label = line.split()
            label = int(label)
            data.append((image_file, label))
        return data

    def __getitem__(self, idx):
        img_name, label = self.XY[idx]
        image = Image.open(self.images_dir + '/' + img_name)
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label
        }


class HCIDataset(OrdinalDataset):
    def __init__(self, config, transform, task, data_splits):
        super().__init__(config, transform, task, data_splits)
        self.X, self.Y = data_splits[task]
        if task == 'val':
            self.X = np.tile(self.X, config.val_boost)
            self.Y = np.tile(self.Y, config.val_boost)

    @staticmethod
    def split_dataset(root_dir, seed=0):
        np.random.seed(seed)
        random.seed(seed)
        classes = {'1930s': 0, '1940s': 1, '1950s': 2, '1960s': 3, '1970s': 4}
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        X_val = []
        Y_val = []
        for c in classes.keys():
            x = glob.glob('{}/{}/*.jpg'.format(root_dir, c))
            y = [classes[c] for _ in range(len(x))]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=50, random_state=seed)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=22, random_state=seed)
            X_train += x_train
            Y_train += y_train
            X_test += x_test
            Y_test += y_test
            X_val += x_val
            Y_val += y_val
        print('size(X_train)={}'.format(len(X_train)))
        print('size(X_test)={}'.format(len(X_test)))
        print('size(X_val)={}'.format(len(X_val)))

        return {
            'train': (X_train, Y_train),
            'test': (X_test, Y_test),
            'val': (X_val, Y_val),
        }

    def get_labels(self):
        return np.array(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = Image.open(self.X[idx])
        image = image.convert('RGB')
        label = self.Y[idx]
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label
        }


class AbaloneDataset(OrdinalDataset):
    def __init__(self, config, transform, task, data_splits):
        super().__init__(config, transform, task, data_splits)
        self.X, self.Y = data_splits[task]

    @staticmethod
    def split_dataset(csv_path, seed=0):
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        X = data[:, :10]
        Y = np.expand_dims(data[:, 10], axis=1)
        Y = Y.astype('int')
        num_classes = int(np.max(Y[:, 0]) + 1)
        Y = Y.astype('float32')
        X = X.astype('float32')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=seed)
        print('size(X_train)={}'.format(len(X_train)))
        print('size(X_test)={}'.format(len(X_test)))
        print('size(X_val)={}'.format(len(X_val)))
        print('num of classes: ', num_classes)
        return {
            'train': (X_train, Y_train),
            'test': (X_test, Y_test),
            'val': (X_val, Y_val),
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'image': self.X[idx],
            'label': self.Y[idx]
        }


class MedMNISTOrdinal(OrdinalDataset):
    def __init__(self, config, transform, task, data_splits):
        super().__init__(config, transform, task, data_splits)
        self.dataset = data_splits[task]
        # self.dataset.montage(length=3, save_folder='.')

    def save_samples(self, k=5, num_classes=5):
        y_count = np.zeros(num_classes)
        samples = []
        for x, y in zip(self.dataset.imgs, self.dataset.labels):
            if y_count[y] <= k:
                samples.append((x, y))
                y_count[y] += 1
                if np.sum(y_count) == k * num_classes: break

        for i, (x, y) in enumerate(samples):
            Image.fromarray(x).save(f'medmnist_class_{y[0] + 1}_{i}.png')

    def get_labels(self):
        return self.dataset.labels

    def __getitem__(self, idx):
        x, y = self.dataset.__getitem__(idx)
        return {
            'image': x,
            'label': y.squeeze()
        }

    def __len__(self):
        return self.dataset.__len__()
