

import sys
sys.path.append("../common")
from training_utils import get_model
import torch
import wandb
from transformers import set_seed
from trainer import FinetuningClassificationTrainer, NoFinetuningClassificationTrainer
from ScriptUtils import parse_testing_args, log_test_results
from data import FinetuningData, PretrainedVectorsData

def init_wandb(args):
    name_of_project = "Moji DFL Testing"

    wandb.init(project=name_of_project, config={
        "architecture": f"{args.model} with linear classifier",
        "finetuning": not args.no_finetuning,
        "dataset": "Twitter race",
        "seed": args.seed,
        "batch size": args.batch_size,
        "training data type": args.training_data,
        "testing data type": args.testing_data,
        "training balancing": args.training_balanced,
        "testing balancing": args.testing_balanced,
        "split type": args.split,
        "optimizer": "adamW",
        "dfl_gamma": args.dfl_gamma,
        "softmax_temperature": args.temp,
        "no_group_labels": args.no_group_labels,
        "control": args.control,
        "few_shot": args.few_shot,
    })

def __main__():
    args = parse_testing_args()
    seed = args.seed
    set_seed(seed)
    init_wandb(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    if args.no_finetuning:
        data = PretrainedVectorsData(
            f"../data/moji/vectors_extracted_from_trained_models/{args.model}/pretrained/vectors_{args.split}_raw_{args.model}_{args.testing_balanced}_128.pt",
            args.seed, None, None, args.few_shot, groups=(0, 1))
    else:
        data = FinetuningData(f"../data/moji/tokens_{args.model}_128.pt",
                              args.seed, args.split, args.testing_balanced, args.few_shot, groups=(0, 1))

    model, checkpoint_folder = get_model(args.no_finetuning, data.n_labels, "../checkpoints/moji/", args.model,
                                         args.seed, args.training_data, args.training_balanced, args.dfl_gamma,
                                         temp=args.temp, no_group_labels=args.no_group_labels, control=args.control)

    loss_fn = torch.nn.CrossEntropyLoss()

    if args.no_finetuning:
        trainer = NoFinetuningClassificationTrainer(model, loss_fn, None, batch_size, [0, 1], device=device)
    else:
        trainer = FinetuningClassificationTrainer(model, loss_fn, None, batch_size, [0, 1], device=device)

    trainer.load_checkpoint(f'{checkpoint_folder}/best_model.pt')

    res = trainer.evaluate(data, args.split)
    log_test_results(res)

if __name__ == "__main__":
    __main__()