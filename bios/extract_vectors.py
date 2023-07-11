import argparse
import random

import torch
from transformers import DebertaTokenizer, BertModel, RobertaModel, DebertaModel, BertTokenizer
from transformers import RobertaTokenizer
import numpy as np
import sys

sys.path.append("../common")
from models import deBERTa_classifier, Bert_classifier
from trainer import load_checkpoint

from data_utils import BiasInBios_extract_vectors_data

N_LABELS = 28

parser = argparse.ArgumentParser(description='Extracting vectors from trained Bias in Bios model.')
parser.add_argument('--model', required=True, type=str, help='the model type',
                    choices=["DFL", "pretrained"])
parser.add_argument('--feature_extractor', type=str,
                    choices=['deberta-base', 'bert-base-uncased'], required=True,
                    help='model to use as a feature extractor')
parser.add_argument('--training_data', type=str, help='type of training data used to train the model',
                    choices=["raw", "scrubbed"], default=None)
parser.add_argument('--training_balanced', type=str, help='balancing of training data used to train the model',
                    choices=["subsampled", "oversampled", "original"], default="original")
parser.add_argument('--seed', '-s', default=0, type=int, help='the random seed the model was trained on')
parser.add_argument('--data', '-d', type=str, help='type of data to extract',
                    choices=["raw", "scrubbed"], default='raw')
parser.add_argument("--dfl_gamma", default=None, type=float, help="Gamma from the DFL loss formulation.")
parser.add_argument("--temp", default=1, type=float, help="Temperature for biased model's softmax.")
parser.add_argument("--no_group_labels", default=False, action="store_true")

args = parser.parse_args()
print("model:", args.model)
print("feature_extractor:", args.feature_extractor)
print("trained on:", args.training_data, args.training_balanced)
print("seed:", args.seed)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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
    elif args.feature_extractor == "roberta-base":
        model = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False, output_attentions=False,
                                             output_hidden_states=False)
    else:
        model = DebertaModel.from_pretrained('microsoft/deberta-base', output_attentions=False,
                                             output_hidden_states=False)

else:
    if args.dfl_gamma > 0:
        if args.no_group_labels:
            load_path = f"../checkpoints/bias_in_bios/{args.feature_extractor}/DFL_no_group_labels/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}/best_model.pt"
        else:
            load_path = f"../checkpoints/bias_in_bios/{args.feature_extractor}/DFL/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}/best_model.pt"
    else:
        load_path = f"../checkpoints/bias_in_bios/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/seed_{args.seed}/best_model.pt"

    if args.feature_extractor == "bert-base-uncased":
        model_ = Bert_classifier(N_LABELS)
    else:
        model_ = deBERTa_classifier(N_LABELS)

    load_checkpoint(model_, load_path)
    model = model_.feature_extractor

if args.model == "pretrained":
    folder = f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}"
elif (args.dfl_gamma is not None) and (args.dfl_gamma > 0):
    if args.no_group_labels:
        folder = f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/DFL_no_group_labels/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}"
    else:
        folder = f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/temp_{args.temp}/seed_{args.seed}"
else:
    folder = f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/gamma_{args.dfl_gamma}/seed_{args.seed}"

model.to(device)

print(f"Going to save to {folder}. Starting extraction...")
data_path = f"../data/biosbias/tokens_{args.data}_{args.feature_extractor}_128.pt"
print(f"Loading tokens from {data_path}")
BiasInBios_extract_vectors_data(args.data, model, feature_extractor=args.feature_extractor, folder=folder,
                                data_path=data_path)
