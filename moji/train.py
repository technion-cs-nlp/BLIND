import sys


sys.path.append("../common")
sys.path.append("../bios")
from training_utils import get_trainer, get_model, init_wandb
from transformers import set_seed
from ScriptUtils import parse_training_args
from data import FinetuningData, PretrainedVectorsData


def get_data(args):
    if args.no_finetuning:
        # data_path = f"../data/moji/vectors_extracted_from_trained_models/{args.model}/pretrained/vectors_{args.data}_{args.model}_128.pt"
        data_train = PretrainedVectorsData(
            f"../data/moji/vectors_extracted_from_trained_models/{args.model}/pretrained/vectors_train_raw_{args.model}_{args.balanced}_128.pt",
            args.seed, None, None, groups=(0, 1))
        data_valid = PretrainedVectorsData(
            f"../data/moji/vectors_extracted_from_trained_models/{args.model}/pretrained/vectors_valid_raw_{args.model}_{args.balanced}_128.pt",
            args.seed, None, None, groups=(0, 1))
    else:
        data_path = f"../data/moji/tokens_{args.model}_128.pt"

        data_train = FinetuningData(data_path, args.seed, "train", args.balanced, groups=(0, 1))
        data_valid = FinetuningData(data_path, args.seed, "valid", args.balanced, groups=(0, 1))

    return data_train, data_valid


def __main__():
    args = parse_training_args()
    init_wandb(args, "Twitter race", "Moji DFL")
    set_seed(args.seed)
    data_train, data_valid = get_data(args)

    checkpoint_folder = "../checkpoints/moji"
    model, checkpoint_folder = get_model(args.no_finetuning, data_train.n_labels, checkpoint_folder, args.model, args.seed, args.data,
                                         args.balanced, args.dfl_gamma, args.temp, args.no_group_labels, args.control)


    trainer = get_trainer(args, model, [0, 1], metric='acc')
    trainer.fit(data_train, data_valid, args.epochs, checkpoint_folder, checkpoint_every=args.checkpointevery,
                print_every=args.printevery)


if __name__ == "__main__":
    __main__()
