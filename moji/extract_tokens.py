import argparse

import torch
from transformers import RobertaTokenizer, BertTokenizer, DebertaTokenizer
import numpy as np

parser = argparse.ArgumentParser(description='Extract tokens from the textual data.')
parser.add_argument('--model', type=str, choices=['deberta-base', 'bert-base-uncased'],
                        help='model to use as a feature extractor')
args = parser.parse_args()

model_version = args.model

if model_version == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(model_version)
elif model_version == 'deberta-base':
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
else:
    print("model is not suppoerted...")
    exit()

y = []
z = []
texts = []
for label in ['neg', 'pos']:
    for aa in ['neg', 'pos']:
        file_name = f'../data/moji/sentiment_race/{label}_{aa}'
        ctr = 0
        with open(file_name, 'rb') as f:
            for line in f.readlines():
                texts.append(line.decode("utf-8", errors='ignore'))
                y.append(1 if label == 'pos' else 0)
                z.append(1 if aa == 'pos' else 0)
                ctr += 1

        print(f"{label} {aa} no. of examples: {ctr}")

max_length = 128
encoded_dict = tokenizer(texts, add_special_tokens=True, padding='max_length', max_length=max_length,
                         truncation=True, return_attention_mask=True)

input_ids = encoded_dict['input_ids']
attention_masks = encoded_dict['attention_mask']

input_ids = np.array(input_ids)
attention_masks = np.array(attention_masks)
y = np.array(y)
z = np.array(z)
path = f"../data/moji/tokens_{model_version}_128.pt"
torch.save({"X": input_ids, "masks": attention_masks, "y": y, "z": z},
           path)
print(f"Saved to {path}")
