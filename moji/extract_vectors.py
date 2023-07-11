import sys
sys.path.append("../common")

from data import FinetuningData

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import tensor
from tqdm import tqdm
from transformers import BertTokenizer, DebertaTokenizer, DebertaModel, BertModel, \
    set_seed

from models import Bert_classifier, deBERTa_classifier
from trainer import load_checkpoint

parser = argparse.ArgumentParser(description='Extracting vectors from trained Deepmoji model.')
parser.add_argument('--model', required=True, type=str, help='the model type',
                    choices=["DFL", "pretrained"])
parser.add_argument('--feature_extractor', default='roberta-base', type=str,
                    choices=['deberta-base', 'bert-base-uncased'],
                    help='model to use as a feature extractor')
parser.add_argument('--training_data', type=str, help='type of training data used to train the model',
                    choices=["raw", "scrubbed"], default="raw")
parser.add_argument('--training_balanced', type=str, help='balancing of training data used to train the model',
                    default="original")
parser.add_argument('--seed', '-s', default=0, type=int, help='the random seed the model was trained on')
parser.add_argument('--data', '-d', type=str, help='type of data to extract',
                    choices=["raw", "scrubbed"], default='raw')
parser.add_argument('--balanced', type=str,
                    help='balancing to extract (relevent on moji because dataset is too large)', default='original')
parser.add_argument("--dfl_gamma", default=0.0, type=float, help="Gamma from the DFL loss formulation.")
parser.add_argument("--temp", default=1.0, type=float, help="Temperature for biased model's softmax.")
parser.add_argument("--no_group_labels", default=False, action="store_true")

args = parser.parse_args()

set_seed(args.seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.feature_extractor == "bert-base-uncased":
    tokenizer = BertTokenizer.from_pretrained(args.feature_extractor)
elif args.feature_extractor == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(args.feature_extractor)
else:
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

if args.model == "pretrained":
    if args.feature_extractor == "bert-base-uncased":
        model = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer=False, output_attentions=False,
                                          output_hidden_states=False)
    elif args.feature_extractor == 'deberta-base':
        model = DebertaModel.from_pretrained('microsoft/deberta-base', output_attentions=False,
                                             output_hidden_states=False)
    else:
        print("model is not suppoerted...")
        exit()
else:
    if args.dfl_gamma > 0:
        if args.no_group_labels:
            load_path = f"../checkpoints/moji/{args.feature_extractor}/DFL_no_group_labels/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}/best_model.pt"
        else:
            load_path = f"../checkpoints/moji/{args.feature_extractor}/DFL/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}/best_model.pt"
    else:
        load_path = f"../checkpoints/moji/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/seed_{args.seed}/best_model.pt"

    if args.feature_extractor == "bert-base-uncased":
        model_ = Bert_classifier(2)
    else:
        model_ = deBERTa_classifier(2)

    load_checkpoint(model_, load_path)
    model = model_.feature_extractor

if args.model == "pretrained":
    folder = f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}"
elif (args.dfl_gamma is not None) and (args.dfl_gamma > 0):
    if args.no_group_labels:
        folder = f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL_no_group_labels/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}"
    else:
        folder = f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}"
else:
    folder = f"../data/moji/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_balanced}/gamma_{args.dfl_gamma}/seed_{args.seed}"

model.to(device)

Path(folder).mkdir(parents=True, exist_ok=True)

print(f"Going to save to {folder}. Starting extraction...")
data_path = f"../data/moji/tokens_{args.feature_extractor}_128.pt"
print(f"Loading tokens from {data_path}")
data = torch.load(data_path)


def extract_vectors_by_split(data_path, seed, split, balanced):
    data_train = FinetuningData(data_path, seed, split, balanced, groups=(0, 1))

    X, y, masks, z = data_train.dataset.tensors
    X, masks = tensor(X).to(device), tensor(masks).to(device)

    vectors = []
    labels = []
    races = []

    with torch.no_grad():
        model.eval()

        for i, x in enumerate(tqdm(X)):
            input_ids = x
            v = model(input_ids.unsqueeze(0), attention_mask=masks[i].unsqueeze(0)).last_hidden_state[:, 0, :][
                0].cpu().detach().numpy()

            vectors.append(v)
            labels.append(y[i])
            races.append(z[i])

        vectors = np.array(vectors)
        labels = np.array(labels)
        races = np.array(races)

    save_path = folder + f"/vectors_{split}_{args.data}_{args.feature_extractor}_{args.balanced}_128.pt"
    torch.save({"X": vectors, "y": labels, "z": races}, save_path)
    print(f"saved to {save_path}")


extract_vectors_by_split(data_path, args.seed, "train", args.balanced)
extract_vectors_by_split(data_path, args.seed, "valid", args.balanced)
extract_vectors_by_split(data_path, args.seed, "test", args.balanced)
