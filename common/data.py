from typing import Union, List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from abc import ABC, abstractmethod

from data_utils import split_and_return_vectors_data, \
    balance_dataset, \
    split_and_return_tokens_data, sample_data_by_ratio


class Data(ABC):
    """
    An abstract class for loading and arranging data from files.
    """

    def __init__(self):
        self.dataset = None


class ClassificationData(Data):
    def __init__(self, path: Union[str, List[str]], seed, split, balanced, few_shot=False, groups=('M', 'F')):
        super().__init__()
        self.groups = groups
        self.dataset = None
        self.original_y = None
        self.original_z = None
        self.n_labels = 0
        self.z = None
        self.code_to_label = None
        self.label_to_code = None
        self.load_dataset(path, seed, split, balanced, few_shot)
        self.perc = self.compute_perc()

    @abstractmethod
    def load_dataset(self, path: str, seed, split, balanced=None, few_shot=0):
        ...

    def compute_perc(self):
        perc = {}
        golden_y = self.original_y
        for label in np.unique(golden_y):
            total_of_label = len(golden_y[golden_y == label])
            indices_subgroup = np.logical_and(golden_y == label, self.original_z == self.groups[1])
            perc_subgroup = len(golden_y[indices_subgroup]) / total_of_label
            perc[label] = perc_subgroup

        return perc


class FinetuningData(ClassificationData):

    def __init__(self, path: Union[str, List[str]], seed, split, balanced, few_shot=0, repeat_indices=None,
                 groups=('M', 'F')):
        self.repeat_indices = repeat_indices
        super().__init__(path, seed, split, balanced, few_shot, groups)

    def split_and_balance(self, path, seed, split, balanced=None, few_shot=0, limit=None, repeat_indices=None):
        data = split_and_return_tokens_data(seed, path)
        cat = data["categories"]
        self.code_to_label = dict(enumerate(cat.categories))
        # self.label_to_code = {v: k for k, v in code_to_label.items()}

        X, y, masks, z = data[split]["X"], data[split]["y"], data[split]["masks"], data[split]["z"]

        if limit is not None:
            X, y, masks, z = X[:limit], y[:limit], masks[:limit], z[:limit]

        if few_shot > 0 and few_shot != 30000:
            # 3000 is a magic number - do nothing
            all_indices = np.random.choice(np.arange(len(y)), size=few_shot)
            X, y, masks, z = X[all_indices], y[all_indices], masks[all_indices], z[all_indices]

        if balanced is not None:
            if balanced in ("oversampled", "subsampled"):
                X, y, z, masks = balance_dataset(X, y, z, masks=masks,
                                                 oversampling=True if balanced == "oversampled" else False,
                                                 groups=self.groups)
            else:
                try:
                    balanced_f = float(balanced)
                    X, y, masks, z = sample_data_by_ratio(X, y, z, balanced_f, 100000, masks=masks,
                                                          groups=self.groups)
                except ValueError:
                    pass

        y = torch.tensor(y).long()
        X = torch.tensor(X).long()
        masks = torch.tensor(masks).long()

        self.z = z
        self.original_z = data[split]["z"]
        self.original_y = data[split]["original_y"]
        self.n_labels = len(np.unique(self.original_y))

        z = pd.Categorical(z).codes
        z = torch.tensor(z).long()

        if repeat_indices is not None:
            X, y, masks, z = torch.cat([X, X[repeat_indices]]), torch.cat([y, y[repeat_indices]]), torch.cat(
                [masks, masks[repeat_indices]]), torch.cat([z, z[repeat_indices]])

        print(
            f"X_{split} shape: {X.shape}, y_{split} shape: {y.shape}, z_{split} shape: {z.shape}")

        return X, y, masks, z

    def load_dataset(self, path: str, seed, split, balanced=None, few_shot=False):
        X, y, masks, z = self.split_and_balance(path, seed, split, balanced, few_shot,
                                                repeat_indices=self.repeat_indices)
        self.dataset = TensorDataset(X, y, masks, z)


class PretrainedVectorsData(ClassificationData):

    def load_dataset(self, path: str, seed, split, balanced=None, few_shot=False):
        data = split_and_return_vectors_data(seed, path, split is not None)
        if split is None:
            split = "train"  # just a hack in case we don't want splitting
        cat = data["categories"]
        self.code_to_label = dict(enumerate(cat.categories))
        X, y = data[split]["X"], data[split]["y"]

        self.z = data[split]["z"]
        self.original_z = data[split]["z"]

        if balanced in ("oversampled", "subsampled"):
            X, y, self.z = balance_dataset(X, y, self.z, oversampling=True if balanced == "oversampled" else False,
                                           groups=self.groups)

        y = torch.tensor(y).long()
        X = torch.tensor(X)

        z = pd.Categorical(self.z).codes
        z = torch.tensor(z).long()

        if few_shot > 0:
            all_indices = []
            for y_ in np.unique(y):
                for z_ in np.unique(z):
                    idx = np.random.choice(torch.where((y == y_) & (z == z_))[0], size=few_shot)
                    all_indices.extend(idx.tolist())
            X, y, z = X[all_indices], y[all_indices], z[all_indices]

        self.dataset = TensorDataset(X, y, z)
        self.original_y = data[split]["original_y"]
        self.n_labels = len(np.unique(self.original_y))
