import sys

# import from project
sys.path.append("../common")
from training_utils import get_model
from trainer import FinetuningClassificationTrainer, NoFinetuningClassificationTrainer
from ScriptUtils import parse_testing_args, log_test_results
from data import FinetuningData, PretrainedVectorsData

# libraries imports
import torch
import wandb
from transformers import set_seed


def init_wandb(args):
    name_of_project = "Bias in Bios DFL Testing"

    if args.model == 'roberta-base':
        model_name = 'RoBERTa'
    else:
        model_name = args.model

    wandb.init(project=name_of_project, config={
        "architecture": f"{model_name} with linear classifier",
        "finetuning": not args.no_finetuning,
        "dataset": "Bias in Bios",
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
        data_path = f"../data/biosbias/vectors_extracted_from_trained_models/{args.model}/pretrained/vectors_{args.testing_data}_{args.model}_128.pt"
        data = PretrainedVectorsData(data_path, args.seed, split=args.split, balanced=args.testing_balanced, few_shot=args.few_shot)
    else:
        data = FinetuningData(f"../data/biosbias/tokens_{args.testing_data}_{args.model}_128.pt",
                              args.seed, args.split, args.testing_balanced, args.few_shot)

    model, checkpoint_folder = get_model(args.no_finetuning, data.n_labels, "../checkpoints/bias_in_bios", args.model,
                                         args.seed, args.training_data, args.training_balanced, args.dfl_gamma,
                                         args.temp, args.no_group_labels, args.control)

    loss_fn = torch.nn.CrossEntropyLoss()

    if args.no_finetuning:
        trainer = NoFinetuningClassificationTrainer(model, loss_fn, None, batch_size, ['M', 'F'], device=device)
    else:
        trainer = FinetuningClassificationTrainer(model, loss_fn, None, batch_size, ['M', 'F'], device=device)

    trainer.load_checkpoint(f'{checkpoint_folder}/best_model.pt')

    res = trainer.evaluate(data, args.split)
    log_test_results(res)

if __name__ == "__main__":
    __main__()
