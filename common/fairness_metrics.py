import json

import numpy as np
import torch
from allennlp.fairness import Independence, Separation, Sufficiency

from ScriptUtils import get_gap_sum
from data import ClassificationData


def compute_gap_metrics(data: ClassificationData, y_pred, groups=('M', 'F')):
    perc = []
    tpr_gap = []
    fpr_gap = []
    precision_gap = []
    F1_gap = []

    golden_y = data.dataset.tensors[1]

    z = data.z

    for label in torch.unique(golden_y):

        group_1_res = compute_tpr_fpr_precision(y_pred, golden_y, z, label.item(), groups[0])
        group_2_res = compute_tpr_fpr_precision(y_pred, golden_y, z, label.item(), groups[1])

        tpr_gap.append(group_2_res["tpr"] - group_1_res["tpr"])
        fpr_gap.append(group_2_res["fpr"] - group_1_res["fpr"])
        precision_gap.append(group_2_res["precision"] - group_1_res["precision"])
        F1_gap.append(group_2_res["f1_score"] - group_1_res["f1_score"])
        perc.append(data.perc[data.code_to_label[label.item()]])

    result = {"tpr_gap": tpr_gap,
              "fpr_gap": fpr_gap,
              "precision_gap": precision_gap,
              "F1_gap": F1_gap,
              "mean abs tpr gap": np.abs(tpr_gap).mean(),
              "mean abs fpr gap": np.abs(fpr_gap).mean(),
              "mean abs f1 gap": np.abs(F1_gap).mean(),
              "mean abs precision gap": np.abs(precision_gap).mean(),
              "perc": perc, "rms": np.sqrt(np.mean(np.array(tpr_gap) ** 2))
              }

    if data.perc is not None:
        result["pearson_tpr_gap"] = np.corrcoef(perc, tpr_gap)[0, 1]
        result["pearson_fpr_gap"] = np.corrcoef(perc, fpr_gap)[0, 1]
        result["pearson_precision_gap"] = np.corrcoef(perc, precision_gap)[0, 1]
        result["pearson_F1_gap"] = np.corrcoef(perc, F1_gap)[0, 1]

    return result


def compute_tpr_fpr_precision(y_pred: torch.Tensor, golden_y, z, label: int, gender: str):
    assert (len(y_pred) == len(golden_y))

    tp_indices = np.logical_and(np.logical_and(z == gender, golden_y.cpu() == label),
                                y_pred.cpu() == label).int()  # only correct predictions of this gender

    n_tp = torch.sum(tp_indices).item()
    pos_indices = np.logical_and(z == gender, golden_y.cpu() == label).int()
    n_pos = torch.sum(pos_indices).item()
    tpr = n_tp / n_pos

    fp_indices = np.logical_and(np.logical_and(z == gender, golden_y.cpu() != label), y_pred.cpu() == label).int()
    neg_indices = np.logical_and(z == gender, golden_y.cpu() != label).int()
    n_fp = torch.sum(fp_indices).item()
    n_neg = torch.sum(neg_indices).item()
    fpr = n_fp / n_neg

    if (n_tp + n_fp == 0):
        precision = 0
    else:
        precision = n_tp / (n_tp + n_fp)

    if precision * tpr == 0:
        f1_score = 0
    else:
        f1_score = 2 * ((precision * tpr) / (precision + tpr))

    return {"tpr": tpr, "fpr": fpr, "precision": precision, "f1_score": f1_score}


def compute_statistical_metrics(data: ClassificationData, y_pred, groups):
    def dictionary_torch_to_number(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                dictionary_torch_to_number(v)
            else:
                d[k] = v.item()

    if len(data.dataset.tensors) == 4:
        z = data.dataset.tensors[3]
    else:
        z = data.dataset.tensors[2]

    y = data.dataset.tensors[1].cpu()
    y_pred = y_pred.cpu()

    independence = Independence(data.n_labels, 2)
    independence(y_pred, z)
    independence_score = independence.get_metric()

    separation = Separation(data.n_labels, 2)
    separation(y_pred, y, z)
    separation_score = separation.get_metric()

    sufficiency = Sufficiency(data.n_labels, 2, dist_metric="kl_divergence")
    sufficiency(y_pred, y, z)
    sufficiency_score = sufficiency.get_metric()

    dictionary_torch_to_number(independence_score)
    dictionary_torch_to_number(separation_score)
    dictionary_torch_to_number(sufficiency_score)

    separation_gaps = [scores[0] - scores[1] for label, scores in
                       sorted(separation_score.items())]  # positive value - more separation for women
    sufficiency_gaps = [scores[0] - scores[1] for label, scores in sorted(sufficiency_score.items())]

    return {"independence": json.dumps(independence_score), "separation": json.dumps(separation_score),
            "sufficiency": json.dumps(sufficiency_score),
            "independence_sum": independence_score[0] + independence_score[1],
            "separation_gap-abs_sum": get_gap_sum(separation_gaps),
            "sufficiency_gap-abs_sum": get_gap_sum(sufficiency_gaps)}


def compute_fairness_metrics(data: ClassificationData, y_pred, groups):
    gap_metrics = compute_gap_metrics(data, y_pred, groups)
    statistical_metrics = compute_statistical_metrics(data, y_pred, groups)
    return {**gap_metrics, **statistical_metrics}
