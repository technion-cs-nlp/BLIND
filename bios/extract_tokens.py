import argparse
from transformers import RobertaTokenizer, DebertaTokenizer, BertTokenizer
import sys
sys.path.append("../common")
from data_utils import BiasInBios_extract_tokens_data

parser = argparse.ArgumentParser(description='Extracting tokens for training bias in bios.')
parser.add_argument('--type', type=str, help='type of training data used to train the model',
                    choices=["raw", "scrubbed"], required=True)
parser.add_argument('--model', type=str, help='model_version for tokenizer', choices=['deberta-base', 'bert-base-uncased'])

args = parser.parse_args()

if args.model == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(args.model)
else:
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

BiasInBios_extract_tokens_data(args.type, tokenizer, args.model)
