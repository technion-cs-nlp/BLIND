import sys

from torch.optim import Adam

sys.path.append("../common")
sys.path.append("../bios")
import torch
import wandb
from torch import nn
from transformers import AdamW
from models import deBERTa_classifier, Bert_classifier
from trainer import FinetuningClassificationTrainer, DFLClassificationTrainer, NoFinetuningClassificationTrainer, \
    DFLNoFinetuningClassificationTrainer


def get_trainer(args, model, groups, metric):
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = args.lr
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.no_finetuning:
        if args.dfl_gamma > 0:
            biased_model = nn.Linear(model.in_features, 2).to(device)
            biased_optimizer = Adam(biased_model.parameters(), lr=args.lr)
            trainer = DFLNoFinetuningClassificationTrainer(model, loss_fn, optimizer, args.batch_size, groups, biased_model,
                                               biased_optimizer, gamma=args.dfl_gamma, temperature=args.temp, device=device)
        else:
            trainer = NoFinetuningClassificationTrainer(model, loss_fn, optimizer, args.batch_size, groups, device=device)
    else:
        if args.dfl_gamma > 0:
            biased_model = nn.Linear(model.classifier.in_features, 2).to(device)
            biased_optimizer = Adam(biased_model.parameters(), lr=1e-3)
            trainer = DFLClassificationTrainer(model, loss_fn, optimizer, args.batch_size, groups, biased_model,
                                               biased_optimizer, gamma=args.dfl_gamma, temperature=args.temp, device=device,
                                               no_group_labels=args.no_group_labels, control=args.control)
        else:
            trainer = FinetuningClassificationTrainer(model, loss_fn, optimizer, args.batch_size, device=device,
                                                      groups=groups, metric=metric)

    return trainer


def get_model(no_finetuning, n_labels, checkpoint_folder_prefix, model_name, seed, data_type, balanced, dfl_gamma, temp,
              no_group_labels, control, fair_batch=None, fairness_type=None):

    if no_group_labels:
        if control:
            folder = 'DFL_no_group_labels_control'
        else:
            folder = 'DFL_no_group_labels'
    else:
        folder = 'DFL'

    if no_finetuning:
        model = torch.nn.Linear(768, n_labels)

        if dfl_gamma > 0:
            checkpoint_folder = f"{checkpoint_folder_prefix}/{model_name}/{folder}_no_finetuning/{data_type}/{balanced}/gamma_{dfl_gamma}/temp_{temp}/seed_{seed}"
        else:
            checkpoint_folder = f"{checkpoint_folder_prefix}/{model_name}/{folder}_no_finetuning/{data_type}/{balanced}/gamma_{dfl_gamma}/seed_{seed}"
    else:
        if model_name == 'roberta-base':
            model = roBERTa_classifier(n_labels)
        if model_name == 'deberta-base':
            model = deBERTa_classifier(n_labels)
        if model_name == 'bert-base-uncased':
            model = Bert_classifier(n_labels)

        if dfl_gamma > 0:
            checkpoint_folder = f"{checkpoint_folder_prefix}/{model_name}/{folder}/{data_type}/{balanced}/gamma_{dfl_gamma}/temp_{temp}/seed_{seed}"
        else:
            folder = 'DFL'
            checkpoint_folder = f"{checkpoint_folder_prefix}/{model_name}/{folder}/{data_type}/{balanced}/gamma_{dfl_gamma}/seed_{seed}"

    print(f"model will be saved to {checkpoint_folder}")
    return model, checkpoint_folder

def init_wandb(args, dataset_name, name_of_project):

    if args.model == 'roberta-base':
        model_name = 'RoBERTa'
    else:
        model_name = args.model

    wandb.init(project=name_of_project, config={
        "learning_rate": args.lr,
        "architecture": f"{model_name} with linear classifier",
        "finetuning": not args.no_finetuning,
        "dataset": dataset_name,
        "seed": args.seed,
        "batch size": args.batch_size,
        "data type": args.data,
        "balancing": args.balanced,
        "total epochs": args.epochs,
        "checkpoint every": args.checkpointevery,
        "optimizer": "adamW",
        "adam_epsilon": args.adam_epsilon,
        "DFL_gamma": args.dfl_gamma,
        "softmax_temperature": args.temp,
        "no_group_labels": args.no_group_labels,
        "control": args.control,
    })