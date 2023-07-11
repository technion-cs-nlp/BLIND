import pickle
import re
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import RobertaTokenizer


def split_and_return_tokens_data(seed, path, other=None, verbose=False):
    data = torch.load(path)
    X, y, att_masks, z = data["X"], data["y"], data["masks"], data["z"]

    cat = pd.Categorical(y)
    y = cat.codes

    if verbose:
        print(f"X shape: {X.shape}, y shape: {y.shape}, att_masks shape: {att_masks.shape}, z shape: {z.shape}")

    other_train = None
    other_valid = None
    other_test = None
    if other is None:
        X_train_valid, X_test, y_train_valid, y_test, att_masks_train_valid, att_masks_test, z_train_valid, z_test, \
        original_y_train_valid, original_y_test = train_test_split(
            X, y, att_masks, z, data["y"], random_state=seed, stratify=y, test_size=0.25)

        X_train, X_valid, y_train, y_valid, att_masks_train, att_masks_valid, z_train, z_valid, original_y_train, \
        original_y_valid = train_test_split(
            X_train_valid, y_train_valid, att_masks_train_valid, z_train_valid, original_y_train_valid,
            random_state=seed, stratify=y_train_valid,
            test_size=0.133)

    else:
        X_train_valid, X_test, y_train_valid, y_test, att_masks_train_valid, att_masks_test, z_train_valid, z_test, other_train_valid, other_test, \
        original_y_train_valid, original_y_test = train_test_split(
            X, y, att_masks, z, other, data["y"], random_state=seed, stratify=y, test_size=0.25)

        X_train, X_valid, y_train, y_valid, att_masks_train, att_masks_valid, z_train, z_valid, other_train, other_valid, \
        original_y_train, original_y_valid = train_test_split(
            X_train_valid, y_train_valid, att_masks_train_valid, z_train_valid, other_train_valid,
            original_y_train_valid, random_state=seed, stratify=y_train_valid,
            test_size=0.133)

    if verbose:
        print(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, att_masks_train shape: {att_masks_train.shape}, z_train shape: {z_train.shape}")
        print(
            f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}, att_masks_valid shape: {att_masks_valid.shape}, z_valid shape: {z_valid.shape}")
        print(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, att_masks_test shape: {att_masks_test.shape}, z_test shape: {z_test.shape}")

    return {
        "categories": cat,
        "train":
            {
                "X": X_train,
                "y": y_train,
                "z": z_train,
                "original_y": original_y_train,
                "masks": att_masks_train,
                "other": other_train
            },
        "test":
            {
                "X": X_test,
                "y": y_test,
                "z": z_test,
                "original_y": original_y_test,
                "masks": att_masks_test,
                "other": other_test
            },
        "valid":
            {
                "X": X_valid,
                "y": y_valid,
                "z": z_valid,
                "original_y": original_y_valid,
                "masks": att_masks_valid,
                "other": other_valid
            }
    }


def split_and_return_vectors_data(seed, path, verbose=False, split=True):
    print(path)
    data = torch.load(path)
    X, y, z = data["X"], data["y"], data["z"]

    # tmp
    # X = torch.load("../data/biosbias/scrubbed_words_vector/data.pt")

    cat = pd.Categorical(y)
    y = cat.codes

    if verbose:
        print(f"X shape: {X.shape}, y shape: {y.shape}, z shape: {z.shape}")

    if split:
        X_train_valid, X_test, y_train_valid, y_test, z_train_valid, z_test, original_y_train_valid, original_y_test = train_test_split(
            X, y, z, data["y"], random_state=seed, stratify=y, test_size=0.25)

        X_train, X_valid, y_train, y_valid, z_train, z_valid, original_y_train, original_y_valid = train_test_split(
            X_train_valid, y_train_valid, z_train_valid, original_y_train_valid, random_state=seed,
            stratify=y_train_valid,
            test_size=0.133)
    else:  # just a hack to avoid splitting in some cases, without changing the entire infrastructure of the code...
        X_train = X
        y_train = y
        z_train = z
        X_valid = X
        y_valid = y
        z_valid = z
        X_test = X
        y_test = y
        z_test = z

    if verbose:
        print(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, z_train shape: {z_train.shape}")
        print(
            f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}, z_valid shape: {z_valid.shape}")
        print(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, z_test shape: {z_test.shape}")

    return {
        "categories": cat,
        "train":
            {
                "X": X_train,
                "y": y_train,
                "z": z_train,
                "original_y": original_y_train
            },
        "test":
            {
                "X": X_test,
                "y": y_test,
                "z": z_test,
                "original_y": original_y_test
            },
        "valid":
            {
                "X": X_valid,
                "y": y_valid,
                "z": z_valid,
                "original_y": original_y_valid
            }
    }


def balance_dataset(X, y, z, masks=None, other=None, oversampling=False, groups=('M', 'F')):
    indexes = []

    for label in np.unique(y):
        female_idx_bool = np.logical_and(y == label, z == groups[1])
        male_idx_bool = np.logical_and(y == label, z == groups[0])
        female_idx = np.arange(len(y))[female_idx_bool]
        male_idx = np.arange(len(y))[male_idx_bool]

        n_female = np.sum(female_idx_bool)
        n_male = np.sum(male_idx_bool)

        if oversampling:
            size = max(n_female, n_male)
            if n_female > n_male:
                sampled_female_idx = np.random.choice(female_idx, size=size, replace=False)
                sampled_male_idx = np.random.choice(male_idx, size=size, replace=True)
            else:
                sampled_female_idx = np.random.choice(female_idx, size=size, replace=True)
                sampled_male_idx = np.random.choice(male_idx, size=size, replace=False)

        else:
            size = min(n_female, n_male)
            sampled_female_idx = np.random.choice(female_idx, size=size, replace=False)
            sampled_male_idx = np.random.choice(male_idx, size=size, replace=False)

        indexes += sampled_female_idx.tolist()
        indexes += sampled_male_idx.tolist()

    other_output = []
    if other is not None:
        for t in other:
            other_output.append(t[indexes])
        if masks is not None:
            return (X[indexes], y[indexes], z[indexes], masks[indexes], *other_output)
        else:
            return (X[indexes], y[indexes], z[indexes], *other_output)
    else:
        if masks is not None:
            return X[indexes], y[indexes], z[indexes], masks[indexes]
        else:
            return X[indexes], y[indexes], z[indexes]


def sample_data_by_ratio(X, y, z, ratio, n, masks=None, groups=(0, 1)):
    happy_aa_idx = np.where(np.logical_and(z == groups[1], y == 1))[0]
    happy_sa_idx = np.where(np.logical_and(z == groups[0], y == 1))[0]
    sad_aa_idx = np.where(np.logical_and(z == groups[1], y == 0))[0]
    sad_sa_idx = np.where(np.logical_and(z == groups[0], y == 0))[0]

    smallest_subset_len1 = min(len(sad_sa_idx), len(happy_aa_idx))
    alternative_n1 = smallest_subset_len1 / (ratio / 2)

    smallest_subset_len2 = min(len(happy_sa_idx), len(sad_aa_idx))
    alternative_n2 = smallest_subset_len2 / ((1 - ratio) / 2)

    n = min(min(n, alternative_n1), alternative_n2)

    n_1 = int(n * ratio / 2)
    n_2 = int(n * (1 - ratio) / 2)

    all_indices = []
    for sub_dataset, amount in [(happy_aa_idx, n_1), (happy_sa_idx, n_2), (sad_aa_idx, n_2), (sad_sa_idx, n_1)]:
        perm = np.random.permutation(len(sub_dataset))
        idx = sub_dataset[perm[:amount]]
        all_indices.extend(idx)

    if masks is not None:
        return X[all_indices], y[all_indices], masks[all_indices], z[all_indices]
    else:
        return X[all_indices], y[all_indices], z[all_indices]


def BiasinBios_create_percentage_data(split_type, seed):
    data = torch.load(f"../data/biosbias/tokens_raw_original_roberta-base_128_{split_type}-seed_{seed}.pt")
    golden_y = data['y']
    z = data['z']
    perc = {}

    for profession in np.unique(golden_y):
        total_of_label = len(golden_y[golden_y == profession])
        indices_female = np.logical_and(golden_y == profession, z == 'F')
        perc_female = len(golden_y[indices_female]) / total_of_label
        perc[profession] = perc_female

    torch.save(perc, f"../data/biosbias/perc_{split_type}-seed_{seed}")


def BiasInBios_extract_tokens_data(datatype, tokenizer, model_name):
    f = open('../data/biosbias/BIOS.pkl', 'rb')
    ds = pickle.load(f)

    labels = []
    genders = []
    inputs = []

    for r in tqdm(ds):
        if datatype == "name":
            sent = " ".join(r['name'])
        elif datatype == "scrubbed":
            sent = r["bio"]  # no start_pos needed
        else:
            sent = r["raw"][r["start_pos"]:]

        inputs.append(sent)
        labels.append(r["title"])
        genders.append(r["gender"])

    if datatype == "name":
        encoded_dict = tokenizer(inputs, add_special_tokens=True, padding=True, return_attention_mask=True)
    else:
        max_length = 128
        encoded_dict = tokenizer(inputs, add_special_tokens=True, padding='max_length', max_length=max_length,
                                 truncation=True, return_attention_mask=True)

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    # Convert the lists into numpy arrays.
    input_ids = np.array(input_ids)
    attention_masks = np.array(attention_masks)
    labels = np.array(labels)
    genders = np.array(genders)
    torch.save({"X": input_ids, "masks": attention_masks, "y": labels, "z": genders},
               f"../data/biosbias/tokens_{datatype}_{model_name}_128.pt")


def BiasInBios_extract_vectors_data(type, model, data_path, feature_extractor='roberta-base',
                                    folder="../data/biosbias"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(data_path)

    vectors = []
    labels = []
    genders = []

    X = torch.tensor(data['X']).to(device)
    y = data['y']
    z = data['z']
    masks = torch.tensor(data['masks']).to(device)

    with torch.no_grad():
        model.eval()

        for i, x in enumerate(tqdm(X)):
            input_ids = x
            v = model(input_ids.unsqueeze(0), attention_mask=masks[i].unsqueeze(0)).last_hidden_state[:, 0, :][
                0].cpu().detach().numpy()

            vectors.append(v)
            labels.append(y[i])
            genders.append(z[i])

        vectors = np.array(vectors)
        labels = np.array(labels)
        genders = np.array(genders)

    Path(folder).mkdir(parents=True, exist_ok=True)
    torch.save({"X": vectors, "y": labels, "z": genders}, folder + f"/vectors_{type}_{feature_extractor}_128.pt")
