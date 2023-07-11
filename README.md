# BLIND

This repo contains the code for [Debiasing NLP Models Without Demographic Information
](https://arxiv.org/abs/2212.10563), which appeared in ACL 2023.
The repo contains code to train models on the two tasks presented in the paper - 
Bios (occupation classification) and Moji (sentiment classification), 
to test the models with all of the fairness metrics from the paper (up to 10 different metrics!),
and to probe the models for any demographic information (sec 5.2 in the paper).

For questions, write to [Hadas Orgad](https://orgadhadas.github.io/).

*** For each one of the scripts, there are many different posslbe configuration parameters, and you can run the script with ``--help`` to see the different options.

*** The scripts run [weights & biases](https://wandb.ai/) for logging training process and test results, so make sure you first configure your wandb account.
## Requirements

Install requirements by

``
pip install requirements.txt
``

## Bias in bios (occupation classification)

### Dataset

You will need to get the dataset following the instructions from the original paper,
[Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting](https://arxiv.org/abs/1901.09451).
(Or write to me - as of the time of writing these lines I have a copy of the dataset).

Change dir to ``bios`` to run the following scripts.

### Training

#### A. extract tokens
Before training, you need to extract the tokens from the dataset with ``extract_tokens.py``:
 
You should have the following file: data/biosbias/BIOS.pkl

And run (example):
``python extract_tokens.py  --type raw --model bert-base-uncased`` 

for without finetuning - need extract vectors. This will also be used to extract vectors for probing.

#### B. running the training script

Training is done with ``train.py``. Example:

 ``python train.py --model bert-base-uncased --data raw --seed 0``

This will train without any debiasing algorithm.

To run with DFL loss (with demographics):

``python train.py --model bert-base-uncased --data raw --seed 0 --no_group_labels --dfl_gamma 16 --temp 1``

**To run BLIND**, just add ``--no_group_labels``:

``python train.py --model bert-base-uncased --data raw --seed 0 --dfl_gamma 16 --temp 1 --no_group_labels ``

