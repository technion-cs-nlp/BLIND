import sys

sys.path.append("../common")
from training_utils import get_trainer, get_model, init_wandb
from transformers import set_seed
from ScriptUtils import parse_training_args, log_test_results
from data import FinetuningData, PretrainedVectorsData


def get_data(args):
    if args.no_finetuning:
        data_path = f"../data/biosbias/vectors_extracted_from_trained_models/{args.model}/pretrained/vectors_{args.data}_{args.model}_128.pt"
        data_train = PretrainedVectorsData(data_path, args.seed, split="train",
                                           balanced=args.balanced)
        data_valid = PretrainedVectorsData(data_path, args.seed, split="valid",
                                           balanced=args.balanced)
    else:
        data_path = f"../data/biosbias/tokens_{args.data}_{args.model}_128.pt"

        data_train = FinetuningData(data_path, args.seed, "train", args.balanced)
        data_valid = FinetuningData(data_path, args.seed, "valid", args.balanced)

    return data_train, data_valid


def __main__():
    args = parse_training_args()
    init_wandb(args, "Bias in Bios", "Bias in Bios DFL")
    set_seed(args.seed)
    data_train, data_valid = get_data(args)

    checkpoint_folder = "../checkpoints/bias_in_bios"
    model, checkpoint_folder = get_model(args.no_finetuning, data_train.n_labels, checkpoint_folder, args.model,
                                         args.seed, args.data, args.balanced, args.dfl_gamma, args.temp,
                                         args.no_group_labels, args.control)

    trainer = get_trainer(args, model, ['M', 'F'], metric='acc')
    trainer.fit(data_train, data_valid, args.epochs, checkpoint_folder, checkpoint_every=args.checkpointevery,
                print_every=args.printevery)

    # validation
    trainer.load_checkpoint(f"{checkpoint_folder}/best_model.pt")
    res = trainer.evaluate(data_valid, "valid")
    log_test_results(res)

if __name__ == "__main__":
    __main__()
