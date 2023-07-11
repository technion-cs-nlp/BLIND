import abc
import argparse
import logging
import os
import pickle
from abc import ABC
from dataclasses import dataclass
from os.path import join
from typing import Callable, Union, Sized
from typing import List
from typing import Tuple

import numpy as np
import torch
import wandb
from sklearn import metrics
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, Subset, IterableDataset
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm, trange
from transformers import set_seed
from transformers.integrations import is_wandb_available

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


def save_checkpoint(output_dir, train_batch_size, num_train_epochs, model, optimizer, loss, epoch, tag='checkpoint'):
    path = os.path.join(output_dir,
                        '{}_b{}_e{}.pt'.format(tag, train_batch_size, num_train_epochs))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(output_dir, train_batch_size, num_train_epochs, model, optimizer, tag='checkpoint') -> Tuple[
    nn.Module, optim.Optimizer, int, float]:
    """
    Load a model from checkpoint for a given output_dir with given arguments.
    model_args must contain train_batch_size and num_train_epochs, and they uniquely identify the model checkpoint.

    Returns: a tuple (model, optimizer, epoch, loss) where model and optimizer are preloaded
    """
    if output_dir is None:
        return model, optimizer, 0, 0.0
    path = os.path.join(output_dir,
                        '{}_b{}_e{}.pt'.format(tag, train_batch_size, num_train_epochs))
    if os.path.exists(path):
        logger.info('Found checkpoint %s, loading model and optimizer...', path)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info('Last checkpoint: loss=%f, epoch=%d', loss, epoch)
        return model, optimizer, epoch, loss
    else:
        logger.info('No checkpoint %s, creating fresh model...', path)
        return model, optimizer, 0, 0.0


@dataclass
class ProbeTrainingArgs:
    train_batch_size: int
    learning_rate: float
    num_train_epochs: int
    checkpoint_steps: int
    early_stopping: int
    early_stopping_tolerance: float


def train_probe(args: ProbeTrainingArgs, train_dataset: Dataset, model: nn.Module,
                dev_dataset=None, loss_fn=nn.CrossEntropyLoss(),
                checkpoint_path=None, collate_fn=None, device=None, verbose=False):


    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    # Training parameters
    probe_classifier = model.to(device)
    criterion = loss_fn.to(device)
    optimizer = optim.AdamW(probe_classifier.parameters(), lr=args.learning_rate)

    loss_list = []
    acc_list = []
    tr_loss = 0.0

    probe_classifier, optimizer, epoch, checkpoint_loss = load_checkpoint(checkpoint_path, args.train_batch_size,
                                                                          args.num_train_epochs, probe_classifier,
                                                                          optimizer)

    if checkpoint_path is not None and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    probe_classifier = probe_classifier.to(device)
    train_iterator = trange(epoch, int(args.num_train_epochs), disable=not verbose)
    epochs_without_improvement = 0
    min_dev_loss = np.inf
    mean_loss = 0

    best_model = None
    for _ in train_iterator:
        acc = 0
        total_size = 0
        mean_loss = 0

        num_batches = 0
        epoch_iterator = tqdm(train_dataloader, desc='Iteration', disable=True)
        probe_classifier.train()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(Variable(t).to(device) for t in batch)
            # Load batch
            #ids = batch[0]
            embedding = batch[0]
            labels = batch[1]

            # Forward propagation
            optimizer.zero_grad()
            outputs = probe_classifier(embedding)


            # Backward propagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate loss and accuracy
            l_item = loss.item()
            _, prediction = torch.max(outputs, 1)
            correct = (prediction == labels).sum().item()
            acc += correct
            tr_loss += l_item
            mean_loss += l_item
            total_size += embedding.size()[0]
            num_batches += 1
            epoch_iterator.set_description('Loss: {}'.format(l_item))

        mean_loss /= num_batches
        epoch += 1
        if epoch % args.checkpoint_steps == 0 and checkpoint_path is not None:
            # Save a checkpoint
            save_checkpoint(checkpoint_path, args.train_batch_size, args.num_train_epochs, probe_classifier, optimizer, mean_loss,
                            epoch)

        acc = acc / total_size
        acc_list.append(acc)
        loss_list.append(mean_loss)

        if dev_dataset is not None:
            # Validation
            probe_classifier.eval()
            dev_loss, _, _ = evaluate_probe(args, dev_dataset, probe_classifier,
                                            loss_fn=loss_fn,
                                            device=device)
            train_iterator.set_description(
                'Accuracy: {:.2f}%, Loss: {:.3f}, Validation Loss: {:.3f}'.format(acc * 100, mean_loss, dev_loss))
            if dev_loss <= min_dev_loss - args.early_stopping_tolerance:
                epochs_without_improvement = 0
                min_dev_loss = dev_loss
                best_model = probe_classifier.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > args.early_stopping:
                    logger.info(f'Early stopping after {epoch} epochs since no validation improvement achieved')
                    break
        else:
            train_iterator.set_description('Accuracy: {:.2f}%, Loss: {:.4f}'.format(acc * 100, mean_loss))

    logger.info(f'Stopped after {epoch}')
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, args.train_batch_size, args.num_train_epochs, probe_classifier, optimizer, mean_loss,
                        epoch, tag='final')
    return acc_list, loss_list


def evaluate_probe(args: ProbeTrainingArgs,
                   eval_dataset: Dataset,
                   model,
                   loss_fn=None,
                   verbose=False,
                   collate_fn=None,
                   device=None):
    """Evaluate a probe model on a given dataset.

    :param args: Arguments for probe training
    :param eval_dataset: The dataset to evaluate on
    :param model: The (trained) probe model
    :param loss_fn: The loss function, Default: CrossEntropyLoss
    :param verbose: If true, prints progress bars and logs. Default: False
    :param collate_fn: A collate function passed to PyTorch `DataLoader`
    :param device:
    :return: A tuple (loss, out_label_ids, preds)
    * loss: mean loss over all samples
    * out_label_ids: The dataset labels, of shape (N,)
    * preds: The model predictions, of shape (N, C), where C is the number of classes that the model predicts
    """
    if eval_dataset is None:
        raise ValueError('eval_dataset cannot be None')
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction='sum')

    tr_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    criterion = loss_fn
    test_sampler = SequentialSampler(eval_dataset)
    test_dataloader = DataLoader(eval_dataset, batch_size=args.train_batch_size,
                                 sampler=test_sampler, collate_fn=collate_fn)

    for batch in tqdm(test_dataloader, desc='Evaluating', disable=not verbose):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            #ids = batch[0]
            embedding = batch[0]
            labels = batch[1]

            outputs = model(embedding)

        loss = criterion(outputs, labels)
        tr_loss += loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = outputs.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    return tr_loss, out_label_ids, preds

class MDLProbe(ABC):
    r"""This class represents a generic probing model which is evaluated using MDL.

    For a concrete implementation, take a look at OnlineCodeMDLProbe
    """

    def __init__(self, model_class: Callable):
        r"""Initialize the probe with a given model class.

        Parameters:
            model_class (Callable): A Callable for building the probe. Assumed to expect no parameters such that a
            probing classifier can be constructed with a call to ``model_class()``

        """
        super().__init__()
        self.model_class = model_class

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        """Evaluate the probe given the MDL metric"""
        raise NotImplementedError()


class OnlineCodeMDLProbe(MDLProbe):
    r"""An MDL probe with Online Coding evaluation.
    Implementation is adapted from Voita and Titov 2020 (https://arxiv.org/pdf/2003.12298.pdf)
    """

    def __init__(self, model_class, fractions: List[float], device=None):
        super().__init__(model_class)
        self.fractions = fractions
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device)

    @staticmethod
    def split_datasets(dataset: Union[Dataset, Sized], fractions: List[float], shuffle=False):
        r"""Split a dataset into portions, given by :fractions:

        Returns a tuple containing 2 lists of size len(fractions) and len(fractions) - 1,
        the first being the train datasets and the latter being the corresponding evaluation (test) datasets.
        The last fraction is always assumed to be 100 (The full dataset).

        Parameters:
            dataset (Dataset): The dataset to split into fractions
            fractions (List[int]): The list of fractions. This should be a monotonically increasing list of integers
            with values between 0 and 100. The last item is assumed to be 100.

        Returns:
            train_portions: A list of Subsets, of size len(fractions)
            eval_portions: A list of Subsets, of size len(fractions) - 1
        """
        if isinstance(Dataset, IterableDataset):
            raise ValueError('dataset must not be of type InstanceDataset, and must implement a __getitem__ method')
        # Normalize to [0, 1]
        fractions = [i / 100 for i in fractions]
        total_len = len(dataset)

        train_portions = []
        eval_portions = []

        if shuffle:
            # In case that the dataset is not ordered randomly, we need to shuffle it
            p = np.random.permutation(total_len)

        for i in range(len(fractions)):

            if shuffle:
                train_subset = Subset(dataset, p[range(0, int(fractions[i] * total_len))])
            else:
                train_subset = Subset(dataset, range(0, int(fractions[i] * total_len)))
            train_portions.append(train_subset)
            if i != len(fractions) - 1:
                # Last dataset does not have a corresponding evaluation set
                if shuffle:
                    eval_subset = Subset(dataset, p[range(int(fractions[i] * total_len), int(fractions[i + 1] * total_len))])
                else:
                    eval_subset = Subset(dataset, range(int(fractions[i] * total_len), int(fractions[i + 1] * total_len)))
                eval_portions.append(eval_subset)

        # torch.save(train_portions, "train_portions.pt")
        # torch.save(eval_portions, "eval_portions.pt")
        return train_portions, eval_portions

    @staticmethod
    def save_report(reporting_root, results: dict):
        r"""Save a report for this probe, after training.

        The report is saved as online_coding.pkl in the given directory.
        """
        pickle.dump(results, open(reporting_root, 'wb'))

    @staticmethod
    def load_report(reporting_root: str):
        r"""Load a saved report for this probe, after it was trained.

        Parameters:
            reporting_root (str): A directory in which the report will be saved.

        Returns:
            report (dict): The saved report, containing two variables: online_coding_list and accuracy.
        """
        return pickle.load(open(reporting_root, 'rb'))

    @staticmethod
    def uniform_code_length(num_classes: int, train_dataset_size: int):
        r"""Calculate the uniform code length for a given training task

        Parameters:
            num_classes (int): Number of classes in the probing (classification) task.
            train_dataset_size (int): The size of the full training dataset which the probe was trained on.

        Returns:
            uniform_code_length (float): The uniform code length for the given training/evaluation parameters of
            the probe.
        """
        return train_dataset_size * np.log2(num_classes)

    @staticmethod
    def online_code_length(num_classes: int, t1: int, losses: List[float]):
        r"""Calculate the online code length.

        Parameters:
            num_classes (int): Number of classes in the probing (classification) task.
            t1 (int): The size of the first training block (fraction) dataset.
            losses (List[float]): The list of (test) losses for each evaluation block (fraction)
            dataset, of size len(fractions).

        Returns:
            online_code_length (float): The online code length for the given training/evaluation parameters of
            the probe.
        """
        return t1 * np.log2(num_classes) + sum(losses)

    def _training_step(self, args: ProbeTrainingArgs,
                       train_ds: Union[Sized, Dataset],
                       dev_ds: Union[Sized, Dataset],
                       eval_ds: Union[Sized, Dataset],
                       collate_fn):
        # Fresh probe instance
        probe = self.model_class().to(self.device)
        loss_fn = self.loss_fn

        acc_list, loss_list = train_probe(args, train_ds, probe,
                                          dev_dataset=dev_ds,
                                          loss_fn=loss_fn,
                                          collate_fn=collate_fn,
                                          device=self.device)
        eval_loss, y_true, preds = evaluate_probe(args, eval_ds, probe,
                                                  device=self.device,
                                                  loss_fn=loss_fn,
                                                  collate_fn=collate_fn)
        return {
            'train': (acc_list, loss_list),
            'eval': (eval_loss, y_true, preds)
        }

    def evaluate(self, train_dataset: Dataset, test_dataset: Union[Dataset, Sized], dev_dataset: Dataset = None,
                 train_batch_size=16, learning_rate=1e-3, num_train_epochs=50,
                 checkpoint_steps=10, early_stopping=4, early_stopping_tolerance=1e-3,
                 reporting_root=None,
                 verbose=False, device=None, collate_fn=None, shuffle=False):
        r"""Evaluate the probe and return the online and uniform code lengths."""

        # torch.save(test_dataset, "test_dataset.pt")
        # torch.save(train_dataset, "train_dataset.pt")

        train_datasets_list, eval_datasets_list = self.split_datasets(train_dataset, self.fractions, shuffle)
        assert len(train_datasets_list) == len(eval_datasets_list) + 1
        # Real training dataset
        train_dataset = train_datasets_list[-1]
        # Online code training fractions
        train_datasets_list = train_datasets_list[:-1]

        online_coding_list = []
        args = ProbeTrainingArgs(
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
            checkpoint_steps=checkpoint_steps,
            early_stopping=early_stopping,
            early_stopping_tolerance=early_stopping_tolerance
        )

        if is_wandb_available():
            wandb.config.update(args.__dict__)

        for index, (train_ds, eval_ds) in tqdm(enumerate(zip(train_datasets_list, eval_datasets_list)),
                                               desc='Online Code training',
                                               total=len(train_datasets_list),
                                               disable=not verbose):
            res = self._training_step(args,
                                      train_ds=train_ds,
                                      dev_ds=dev_dataset,
                                      eval_ds=eval_ds,
                                      collate_fn=collate_fn)
            fraction = self.fractions[index]
            eval_loss = res['eval'][0]
            online_coding_list.append({
                'fraction': fraction,
                'eval_loss': eval_loss,
                'train_acc_list': res['train'][0],
                'train_loss_list': res['train'][1]
            })
            if is_wandb_available():
                wandb.log({
                    'loss': eval_loss,
                    'fraction': fraction
                })

        res = self._training_step(args, train_dataset, dev_dataset, test_dataset, collate_fn)

        eval_loss, y_true, preds = res['eval']
        y_pred = np.argmax(preds, axis=1)
        correct = (y_true == y_pred).sum()
        accuracy = correct / len(test_dataset)

        # save results
        self.save_report(reporting_root, {
            'online_coding_list': online_coding_list,
            'training': {
                'eval_loss': eval_loss,
                'train_acc_list': res['train'][0],
                'train_loss_list': res['train'][1],
                'accuracy': accuracy
            },
            'eval': {
                'classification_report': metrics.classification_report(y_true, y_pred)
            }
        })

        num_classes = len(np.unique(y_pred))
        train_dataset_size = len(train_dataset)
        uniform_cdl = self.uniform_code_length(num_classes, train_dataset_size)
        online_cdl = self.online_code_length(num_classes, len(train_datasets_list[0]),
                                             list(map(lambda obj: obj['eval_loss'], online_coding_list)))

        if is_wandb_available():
            wandb.run.summary['eval_loss'] = eval_loss
            wandb.run.summary['uniform_cdl'] = uniform_cdl
            wandb.run.summary['online_cdl'] = online_cdl
            wandb.run.summary['compression'] = round(uniform_cdl / online_cdl, 2)
            wandb.run.summary['eval_accuracy'] = accuracy
        return uniform_cdl, online_cdl

@dataclass
class OnlineCodingExperimentResults:
    name: str
    uniform_cdl: float
    online_cdl: float
    compression: float
    report: dict
    fractions: List[float]

def general_probing_args(parser):
    parser.add_argument('--seed', type=int, help='the random seed to check on', required=True)
    parser.add_argument('--model_seed', type=int, help='the random seed used to train the model', required=True)
    parser.add_argument('--embedding_size', type=int, help='embedding size', default=768)
    parser.add_argument('--batch_size', type=int, help='batch size to train the probe', default=16)
    parser.add_argument('--probe_type', type=str, help='linear probe or MLP', choices=['linear', 'mlp'])

def general_MDL_args():
    parser = argparse.ArgumentParser(description='Probe vectors for gender of example using MDL probes.')
    general_probing_args(parser)
    parser.add_argument('--mdl_fractions', nargs='+', type=int, help='linear probe of MLP',
                        default=[2.0, 3.0, 4.4, 6.5, 9.5, 14.0, 21.0, 31.0, 45.7, 67.6, 100])

    return parser

def build_probe(input_size, num_classes=2, probe_type='mlp'):
    probes = {
        'mlp': lambda: nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, num_classes)
        ),
        'linear': lambda: nn.Linear(input_size, num_classes)
    }
    return probes[probe_type]()

def create_probe(args):
    return build_probe(args.embedding_size, probe_type=args.probe_type)

def run_MDL_probing(args, load_fn, task_name, shuffle):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online_code_probe = OnlineCodeMDLProbe(lambda: create_probe(args), args.mdl_fractions, device=device)

    train_dataset, val_dataset, test_dataset = load_fn(args)
    train_dataset = train_dataset

    reporting_root = join(os.getcwd(), f'mdl_results/online_coding_{task_name}.pkl')

    uniform_cdl, online_cdl = online_code_probe.evaluate(train_dataset, test_dataset, val_dataset,
                                                         reporting_root=reporting_root, verbose=True, device=device,
                                                         train_batch_size=args.batch_size, shuffle=shuffle)
    compression = round(uniform_cdl / online_cdl, 2)
    report = online_code_probe.load_report(reporting_root)

    exp_results = OnlineCodingExperimentResults(
        name=task_name,
        uniform_cdl=uniform_cdl,
        online_cdl=online_cdl,
        compression=compression,
        report=report,
        fractions=args.mdl_fractions
    )

    return exp_results