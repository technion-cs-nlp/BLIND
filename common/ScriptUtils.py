import argparse
import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset

from data import PretrainedVectorsData


def parse_training_args():
    parser = argparse.ArgumentParser(description='Run finetuning training process on Bias in Bios dataset.')
    parser.add_argument('--model', default='bert-base-uncased', type=str, choices=['deberta-base', 'bert-base-uncased'],
                        help='model to use as a feature extractor')
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='the batch size for the training process')
    parser.add_argument('--data', '-d', required=False, type=str, help='the data type to train on',
                        choices=["raw", "scrubbed"], default='raw')
    parser.add_argument('--balanced', help='balancing of the training data', default="original")
    parser.add_argument('--lr', default=5e-5, type=float, help='the learning rate')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='the number of epochs')
    parser.add_argument('--seed', '-s', default=0, type=int, help='the random seed')
    parser.add_argument('--printevery', '-pe', default=1, type=int, help='print results every this number of epochs')
    parser.add_argument('--checkpointevery', '-ce', default=11, type=int,
                        help='print results every this number of epochs')
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--dfl_gamma", default=0.0, type=float, help="Gamma from the DFL loss formulation.")
    parser.add_argument("--temp", default=1.0, type=float, help="Temperature for biased model's softmax.")
    parser.add_argument("--no_finetuning", default=False, action="store_true", help="Use this if you want the feature"
                                                                                    "extractor frozen and only the"
                                                                                    " linear layer trained")
    parser.add_argument("--no_group_labels", default=False, action="store_true", help="Activate BLIND - no demographics")
    parser.add_argument("--control", default=False, action="store_true", help="Control run (Control in the paper)")
    args = parser.parse_args()

    print("Batch size:", args.batch_size)
    print("Data type:", args.data)
    print("Balancing:", args.balanced)
    print("Learning rate:", args.lr)
    print("Number of epochs:", args.epochs)
    print("Random seed:", args.seed)
    print("Print Every:", args.printevery)
    print("Checkpoint Every:", args.checkpointevery)

    return args


def parse_testing_args():
    parser = argparse.ArgumentParser(description='Run testing on Bias in Bios dataset.')
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='the batch size to test with')
    parser.add_argument('--training_data', required=False, type=str, help='the data type the model was trained on',
                        choices=["raw", "scrubbed"], default="raw")
    parser.add_argument('--testing_data', required=False, type=str, help='the data type to test on',
                        choices=["raw", "scrubbed"], default="raw")
    parser.add_argument('--training_balanced', type=str, help='balancing of the training data', default="original")
    parser.add_argument('--testing_balanced', type=str, help='balancing of the test data', default="original")
    parser.add_argument('--split', type=str, help='the split of the dataset to test on',
                        choices=["train", "test", "valid"], default="test")
    parser.add_argument('--seed', '-s', default=0, type=int, help='the random seed')
    parser.add_argument('--model', default='bert-base-uncased', type=str, choices=['deberta-base', 'bert-base-uncased'],
                        help='model to use as a feature extractor')
    parser.add_argument("--dfl_gamma", default=0.0, type=float, help="Gamma from the DFL loss formulation.")
    parser.add_argument("--temp", default=1.0, type=float, help="Temperature for biased model's softmax.")
    parser.add_argument("--no_finetuning", default=False, action="store_true", help="Use this if you want the feature"
                                                                                    "extractor frozen and only the"
                                                                                    " linear layer trained")
    parser.add_argument("--no_group_labels", default=False, action="store_true", help="Activate BLIND - no demographics")
    parser.add_argument("--control", default=False, action="store_true", help="Control run (Control in the paper)")
    args = parser.parse_args()
    return args


def preprocess_probing_data(X, z):
    z[z == 'F'] = 1
    z[z == 'M'] = 0
    z = z.astype(int)
    X, z = X, torch.tensor(z).long()

    return TensorDataset(X, z)


def load_probing_dataset(path):
    data = torch.load(path)

    ds = preprocess_probing_data(torch.tensor(data['X']), data['z'])
    return ds


def load_bias_in_bios_vectors(args):
    seed = args.seed
    model_seed = args.model_seed
    # set_seed(args.seed)

    if args.model == "DFL":
        if args.dfl_gamma > 0:
            if args.no_group_labels:
                data_train = PretrainedVectorsData(
                    f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL_no_group_labels/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                    seed, "train", args.testing_balanced)
                data_valid = PretrainedVectorsData(
                    f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL_no_group_labels/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                    seed, "valid", args.testing_balanced)
                data_test = PretrainedVectorsData(
                    f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL_no_group_labels/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                    seed, "test", args.testing_balanced)
            else:
                data_train = PretrainedVectorsData(
                    f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                    seed, "train", args.testing_balanced)
                data_valid = PretrainedVectorsData(
                    f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                    seed, "valid", args.testing_balanced)
                data_test = PretrainedVectorsData(
                    f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                    seed, "test", args.testing_balanced)
        else:
            data_train = PretrainedVectorsData(
                f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                seed, "train", args.testing_balanced)
            data_valid = PretrainedVectorsData(
                f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                seed, "valid", args.testing_balanced)
            data_test = PretrainedVectorsData(
                f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                seed, "test", args.testing_balanced)
    elif args.model == "random":
        data_train = PretrainedVectorsData(
            f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/random/seed_{seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
            seed, "train", args.testing_balanced)
        data_valid = PretrainedVectorsData(
            f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/random/seed_{seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
            seed, "valid", args.testing_balanced)
        data_test = PretrainedVectorsData(
            f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/random/seed_{seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
            seed, "test", args.testing_balanced)
    else:
        data_train = PretrainedVectorsData(f"../data/biosbias/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                           seed, "train", args.testing_balanced)
        data_valid = PretrainedVectorsData(f"../data/biosbias/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                           seed, "valid", args.testing_balanced)
        data_test = PretrainedVectorsData(f"../data/biosbias/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                          seed, "valid", args.testing_balanced)

    return preprocess_probing_data(data_train.dataset.tensors[0], data_train.z), \
           preprocess_probing_data(data_valid.dataset.tensors[0], data_valid.z), \
           preprocess_probing_data(data_test.dataset.tensors[0], data_test.z)

def load_moji_vectors(args):

    seed = args.seed
    model_seed = args.model_seed

    if args.model == "DFL":
        if args.dfl_gamma > 0:
            if args.no_group_labels:
                data_train = PretrainedVectorsData(
                    f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL_no_group_labels/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}/vectors_train_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt",
                    args.seed, None, None, groups=(0, 1))
                data_valid = PretrainedVectorsData(
                    f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL_no_group_labels/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}/vectors_valid_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt",
                    args.seed, None, None, groups=(0, 1))
                data_test = PretrainedVectorsData(
                    f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL_no_group_labels/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}/vectors_test_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt",
                    args.seed, None, None, groups=(0, 1))
            else:
                data_train = PretrainedVectorsData(
                    f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}/vectors_train_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt",
                    args.seed, None, None, groups=(0, 1))
                data_valid = PretrainedVectorsData(
                    f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}/vectors_valid_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt",
                    args.seed, None, None, groups=(0, 1))
                data_test = PretrainedVectorsData(
                    f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}/vectors_test_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt",
                    args.seed, None, None, groups=(0, 1))
        else:
            data_train = PretrainedVectorsData(
                f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL/{args.training_data}/{args.training_balanced}/gamma_0.0/seed_{args.seed}/vectors_train_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt",
                args.seed, None, None, groups=(0, 1))
            data_valid = PretrainedVectorsData(
                f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL/{args.training_data}/{args.training_balanced}/gamma_0.0/seed_{args.seed}/vectors_valid_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt",
                args.seed, None, None, groups=(0, 1))
            data_test = PretrainedVectorsData(
                f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL/{args.training_data}/{args.training_balanced}/gamma_0.0/seed_{args.seed}/vectors_test_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt",
                args.seed, None, None, groups=(0, 1))
    elif args.model == "random":
        data_train = PretrainedVectorsData(
            f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/random/seed_{seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
            seed, "train", args.testing_balanced)
        data_valid = PretrainedVectorsData(
            f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/random/seed_{seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
            seed, "valid", args.testing_balanced)
        data_test = PretrainedVectorsData(
            f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/random/seed_{seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
            seed, "test", args.testing_balanced)
    else:
        data_train = PretrainedVectorsData(
            f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/vectors_train_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt", args.seed, None, None, groups=(0, 1))
        data_valid = PretrainedVectorsData(
            f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/vectors_valid_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt", args.seed, None, None, groups=(0, 1))
        data_test = PretrainedVectorsData(
            f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/vectors_test_raw_{args.feature_extractor}_{args.testing_balanced}_128.pt", args.seed, None, None, groups=(0, 1))

    return TensorDataset(data_train.dataset.tensors[0], torch.tensor(data_train.z).long()),\
           TensorDataset(data_valid.dataset.tensors[0], torch.tensor(data_valid.z).long()),\
           TensorDataset(data_test.dataset.tensors[0], torch.tensor(data_test.z).long())

def get_avg_gap(gap):
    gap = np.array(gap)
    f = np.mean(gap[gap > 0])
    m = -np.mean(gap[gap < 0])
    return {"f": f, "m": m}


def get_gap_sum(gap):
    return np.abs(np.array(gap)).sum()


def log_test_results(res):
    wandb.run.summary[f"acc"] = res['acc']

    if 'loss' in res:
        wandb.run.summary[f"avg_loss"] = res['loss']

    perc = res['perc']

    # gaps
    wandb.run.summary[f"tpr_gap-pearson"] = res['pearson_tpr_gap']
    wandb.run.summary[f"tpr_gap-abs_sum"] = get_gap_sum(res['tpr_gap'])
    table_data = [[x, y] for (x, y) in zip(perc, res['tpr_gap'])]
    table = wandb.Table(data=table_data, columns=["perc of females", "tpr gap"])
    wandb.log({f"tpr_gap_chart": wandb.plot.line(table, "perc of females", "tpr gap",
                                                 title=f"tpr gap chart")})

    wandb.run.summary[f"fpr_gap-pearson"] = res['pearson_fpr_gap']
    wandb.run.summary[f"fpr_gap-abs_sum"] = get_gap_sum(res['fpr_gap'])
    table_data = [[x, y] for (x, y) in zip(perc, res['fpr_gap'])]
    table = wandb.Table(data=table_data, columns=["perc of females", "fpr gap"])
    wandb.log({f"fpr_gap_chart": wandb.plot.line(table, "perc of females", "fpr gap",
                                                 title=f"fpr gap chart")})

    wandb.run.summary[f"precision_gap-pearson"] = res['pearson_precision_gap']
    wandb.run.summary[f"precision_gap-abs_sum"] = get_gap_sum(res['precision_gap'])
    table_data = [[x, y] for (x, y) in zip(perc, res['precision_gap'])]
    table = wandb.Table(data=table_data, columns=["perc of females", "precision gap"])
    wandb.log({f"precision_gap_chart": wandb.plot.line(table, "perc of females", "precision gap",
                                                       title=f"precision gap chart")})

    # Allennlp metrics

    ## independence
    wandb.run.summary['independence'] = res['independence']
    wandb.run.summary['independence-sum'] = res['independence_sum']

    ## separation
    wandb.run.summary['separation'] = res['separation']
    wandb.run.summary['separation_gap-abs_sum'] = res['separation_gap-abs_sum']

    ## sufficiency
    wandb.run.summary['sufficiency'] = res['sufficiency']
    wandb.run.summary['sufficiency_gap-abs_sum'] = res['sufficiency_gap-abs_sum']

    wandb.run.summary['rms'] = res['rms']
