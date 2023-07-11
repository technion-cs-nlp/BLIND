import copy
import os
import sys
from abc import abstractmethod, ABC
from pathlib import Path
from typing import NamedTuple

from fairness_metrics import compute_gap_metrics, compute_statistical_metrics, compute_fairness_metrics

sys.path.append("../common")
from loss import FocalLoss

import json
import numpy as np
import torch
import wandb
from data import Data
from allennlp.fairness import Independence, Separation, Sufficiency
from torch.utils.data import DataLoader
from tqdm import tqdm


class BestModel(NamedTuple):
    epoch: int
    result: dict
    model: torch.nn.Module


def load_checkpoint(model, load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path)
    print(f'Model loaded from <== {load_path}')

    if callable(state_dict['model_state_dict']):
        model.load_state_dict(state_dict['model_state_dict']())
    else:
        model.load_state_dict(state_dict['model_state_dict'])


class Trainer(ABC):
    """
        A class abstracting the various tasks of training models.

        Provides methods at multiple levels of granularity:
        - Multiple epochs (fit)
        - Single epoch (train_epoch/test_epoch)
        - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, bsz, scheduler=None, device='cpu', metric='acc'):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.verbose = False
        self.best_model: BestModel = None
        self.metric = metric
        self.batch_size = bsz

        # move to device
        self.model.to(self.device)

    def fit(self, train_data, val_data, num_epochs, checkpoint_folder, print_every=1,
            checkpoint_every=1, sampler=None):

        for epoch in range(1, num_epochs + 1):
            self.verbose = ((epoch % print_every) == 0 or (epoch == num_epochs - 1))
            self._print(f'--- EPOCH {epoch}/{num_epochs} ---', self.verbose)

            self.model.train()
            train_result = self.train_epoch(train_data, sampler)
            self.model.eval()
            valid_result = self.evaluate(val_data, "valid")

            is_best = self.best_checkpoint(valid_result, epoch)
            self.save_checkpoint(checkpoint_folder, epoch, valid_result, checkpoint_every, save_best=is_best)

            # print(train_result, valid_result)
            self.log_epoch_results(train_result, valid_result)

        # self.save_checkpoint(checkpoint_folder, None, None, save_best=True)

    def train_batch(self, batch):

        self.optimizer.zero_grad()
        res = self.forward_fn(batch)
        loss = res['loss']
        loss.backward()
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return res

    def save_checkpoint(self, save_folder, epoch, valid_result, checkpoint_every, save_best):

        if save_folder == None:
            return

        Path(save_folder).mkdir(parents=True, exist_ok=True)

        # save_path = None
        if (epoch % checkpoint_every) == 0:
            save_path = f"{save_folder}/ckpt_epoc_{epoch}.pt"
            torch.save({'model_state_dict': self.model.state_dict(),
                        'epoch_data': valid_result,
                        'epoch': epoch}, save_path)
            print(f'Model saved to ==> {save_path}')
        if save_best:
            # best_model_path = f"{save_folder}/ckpt_epoc_{self.best_model.epoch}.pt"
            save_path = f"{save_folder}/best_model.pt"
            # os.system(f"cp {best_model_path} {save_path}")
            # torch.save({'model_state_dict': self.best_model.state_dict,
            #             'epoch_data': self.best_model.result,
            #             'epoch': self.best_model.epoch}, save_path)
            torch.save({'model_state_dict': self.model.state_dict(),
                        'epoch_data': valid_result,
                        'epoch': epoch}, save_path)
            print(f'Model saved to ==> {save_path}')

        # if save_path is not None:
        #     torch.save({'model_state_dict': self.model.state_dict(),
        #                 'epoch_data': valid_result,
        #                 'epoch': epoch}, save_path)
        #     print(f'Model saved to ==> {save_path}')

    def load_checkpoint(self, load_path):

        if load_path == None:
            return

        state_dict = torch.load(load_path)
        print(f'Model loaded from <== {load_path}')

        if callable(state_dict['model_state_dict']):
            self.model.load_state_dict(state_dict['model_state_dict']())
        else:
            self.model.load_state_dict(state_dict['model_state_dict'])

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    def best_checkpoint(self, valid_result, epoch):

        if self.metric == 'acc':
            indicator = (not self.best_model) or self.best_model.result[self.metric] < valid_result[self.metric]
        else:
            indicator = (not self.best_model) or self.best_model.result[self.metric] > valid_result[self.metric]

        if indicator:
            self.best_model = BestModel(epoch, valid_result.copy(), copy.deepcopy(self.model))
            wandb.run.summary["best_metric"] = valid_result[self.metric]
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["metric"] = self.metric
            return True
        else:
            return False

    @abstractmethod
    def forward_fn(self, batch):
        ...

    @abstractmethod
    def evaluate(self, data, split_type: str):
        ...

    @abstractmethod
    def log_epoch_results(self, train_result, valid_result):
        ...

    @abstractmethod
    def train_epoch(self, train_data, sampler):
        """
                Train once over a training set (single epoch).
                :param train_data: the training data object
                :return: An epoch result dictionary.
                """
        ...


class ClassificationTrainer(Trainer, ABC):

    def __init__(self, model, loss_fn, optimizer, bsz, groups, scheduler=None, device='cpu', metric='acc'):

        super().__init__(model, loss_fn, optimizer, bsz, scheduler=scheduler, device=device, metric=metric)
        self.groups = groups

    def log_epoch_results(self, train_result, valid_result):

        train_result_new = {}
        for k in train_result:
            train_result_new[f"train_{k}"] = train_result[k]
        valid_result_new = {}
        for k in valid_result:
            valid_result_new[f"valid_{k}"] = valid_result[k]

        wandb.log({**train_result_new, **valid_result_new})

    def evaluate(self, data: Data, split_type: str):
        dl = DataLoader(data.dataset, batch_size=self.batch_size, shuffle=False)

        y_pred, logits = self.predict(dl)
        y = data.dataset.tensors[1].to(self.device)
        loss = self.loss_fn(logits, y)
        total_correct = torch.sum(y == logits.argmax(dim=1)).item()

        total_examples = len(y)
        accuracy = total_correct / total_examples
        fairness_metrics_result = compute_fairness_metrics(data, y_pred, self.groups)

        return {"loss": loss, "acc": accuracy, **fairness_metrics_result}

    def predict(self, data: DataLoader):
        self.model.eval()
        all_y_pred = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(data):
                logits = self.forward_fn(batch)['logits']
                y_pred = logits.argmax(dim=1)
                all_y_pred.append(y_pred)
                all_logits.append(logits)

        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        return all_y_pred, all_logits

    def train_epoch(self, train_data: Data, sampler):
        losses = []
        total_correct = 0
        total_examples = 0

        if sampler is not None:
            train_iter = DataLoader(train_data.dataset, batch_size=1, sampler=sampler)
        else:
            train_iter = DataLoader(train_data.dataset, batch_size=self.batch_size, shuffle=True)

        for batch in tqdm(train_iter):
            y = batch[1]
            batch_res = self.train_batch(batch)
            losses.append(batch_res['loss'].item())
            total_correct += batch_res['n_correct']
            total_examples += len(y)

        accuracy = total_correct / total_examples
        loss = np.mean(losses)

        return {"loss": loss, "acc": accuracy}


class FinetuningClassificationTrainer(ClassificationTrainer):

    def forward_fn(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        X, y, att_mask, _ = batch

        X = X.squeeze()
        att_mask = att_mask.squeeze()
        y = y.squeeze()
        logits = self.model.forward(X, att_mask)  # shape (batch_size, n_labels)
        loss = self.loss_fn(logits, y)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}


class DFLClassificationTrainer(ClassificationTrainer):

    def __init__(self, model, loss_fn, optimizer, bsz, groups, biased_model, biased_optimizer, gamma=2, alpha=1,
                 temperature=1,
                 scheduler=None, device='cpu', no_group_labels=False, control=False):
        super().__init__(model, loss_fn, optimizer, bsz, groups, scheduler, device)
        self.dfl_loss_fn = FocalLoss(gamma=gamma, alpha=alpha, temperature=temperature)
        self.biased_model = biased_model
        self.biased_optimizer = biased_optimizer
        self.no_group_labels = no_group_labels
        self.control = control

    def forward_fn(self, batch, is_validation=False):
        batch = tuple(t.to(self.device) for t in batch)
        X, y, att_mask, z = batch

        logits, features = self.model.forward(X, att_mask, return_features=True)
        features_np = features.detach().cpu().numpy()
        z_np = z.cpu().numpy()

        # just a hack to see if we are on train mode
        if self.model.training:
            # self.biased_model.partial_fit(features_np, z_np.ravel(), np.arange(2))
            self.biased_optimizer.zero_grad()
            logits_b = self.biased_model(features.detach())

            if self.no_group_labels:
                y_pred = logits.argmax(dim=1)
                if not self.control:
                    y_acc = torch.where(y == y_pred, 1, 0)
                else:
                    y_acc = torch.randint(0, 2, y_pred.shape).to(self.device)
                biased_loss = self.loss_fn(logits_b, y_acc)
            else:
                biased_loss = self.loss_fn(logits_b, z)
            biased_loss.backward()
            self.biased_optimizer.step()

        with torch.no_grad():
            logits_b = self.biased_model(features).detach()

        loss = self.dfl_loss_fn(logits, y, logits_b, z)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}

    def save_checkpoint(self, save_folder, epoch, valid_result, checkpoint_every, save_best):
        if save_best:
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            save_path = f"{save_folder}/biased_model.pt"
            torch.save(self.biased_model.state_dict(), save_path)
            print(f'Biased model saved to ==> {save_path}')
        super().save_checkpoint(save_folder, epoch, valid_result, checkpoint_every, save_best)


class NoFinetuningClassificationTrainer(ClassificationTrainer):

    def forward_fn(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        X, y, z = batch  # X shape (bsz, hidden_dim)

        logits = self.model.forward(X)  # shape (bsz, n_labels)
        loss = self.loss_fn(logits, y)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}


class DFLNoFinetuningClassificationTrainer(ClassificationTrainer):

    def __init__(self, model, loss_fn, optimizer, bsz, groups, biased_model, biased_optimizer, gamma=2, alpha=1,
                 temperature=1,
                 scheduler=None, device='cpu'):
        super().__init__(model, loss_fn, optimizer, bsz, groups, scheduler, device)
        self.dfl_loss_fn = FocalLoss(gamma=gamma, alpha=alpha, temperature=temperature)
        self.biased_model = biased_model
        self.biased_optimizer = biased_optimizer

    def forward_fn(self, batch, is_validation=False):
        batch = tuple(t.to(self.device) for t in batch)
        X, y, z = batch

        logits = self.model.forward(X)

        # just a hack to see if we are on train mode
        if self.model.training:
            self.biased_optimizer.zero_grad()
            logits_b = self.biased_model(X.detach())
            biased_loss = self.loss_fn(logits_b, z)
            biased_loss.backward()
            self.biased_optimizer.step()

        with torch.no_grad():
            logits_b = self.biased_model(X).detach()

        loss = self.dfl_loss_fn(logits, y, logits_b, z)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        total_correct = torch.sum(z == logits_b.argmax(dim=1)).item()
        total_examples = len(z)
        biased_model_acc = total_correct / total_examples
        wandb.log({'biased_model_acc': biased_model_acc})

        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}
