import sys

import wandb

sys.path.append("../common")

from mdl import run_MDL_probing, general_MDL_args

sys.path.append('../src')
from ScriptUtils import load_bias_in_bios_vectors

def parse_args():
    parser = general_MDL_args()
    parser.add_argument('--model', required=True, type=str, help='the model type',
                        choices=["DFL"])
    parser.add_argument('--feature_extractor', default='bert-base-uncased', type=str, choices=['bert-base-uncased','deberta-base'],
                        help='model to use as a feature extractor')
    parser.add_argument('--training_data', required=False, type=str, help='the data type the model was trained on',
                        choices=["raw", "scrubbed"], default=None)
    parser.add_argument('--training_balanced', type=str, help='balancing of the training data',
                        choices=["subsampled", "oversampled", "original"], default="original")
    parser.add_argument('--type', type=str, help='the type of vectors to probe',
                        choices=["raw", "scrubbed"])
    parser.add_argument('--testing_balanced', type=str, help='balancing of the testing data',
                        choices=["subsampled", "oversampled", "original"], default="original")
    parser.add_argument("--dfl_gamma", default=None, type=float, help="Gamma from the DFL loss formulation.")
    parser.add_argument("--temp", default=1.0, type=float, help="Temperature for biased model's softmax.")
    parser.add_argument("--no_group_labels", default=False, action="store_true")

    args = parser.parse_args()
    print(args)
    return args


def init_wandb(args):
    if args.feature_extractor == 'roberta-base':
        model_name = 'RoBERTa'
    else:
        model_name = args.feature_extractor

    wandb.init(project="Bias in Bios DFL - MDL probing", config={
        "architecture": f"{model_name} with linear classifier",
        "seed": args.seed,
        "model_seed": args.model_seed,
        "training data": args.training_data,
        "training balancing": args.training_balanced,
        "testing balancing": args.testing_balanced,
        "type": args.type,
        "model": args.model,
        "dfl_gamma": args.dfl_gamma,
        "softmax_temperature": args.temp,
        "no_group_labels": args.no_group_labels
    })


def main():
    args = parse_args()
    init_wandb(args)

    task_name = f'biasinbios_model_{args.model}_type_{args.type}_{args.testing_balanced}_training_{args.training_data}_{args.training_balanced}_seed_{args.seed}'
    run_MDL_probing(args, load_bias_in_bios_vectors, task_name, shuffle=True)

if __name__ == '__main__':
    main()