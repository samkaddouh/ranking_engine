import numpy as np


def dcg(scores):
    scores = np.array(scores)
    return np.sum((2 ** scores - 1) / np.log2(np.arange(2, len(scores) + 2)))


def ndcg_at_k(y_true, y_score, k=10):
    order = np.argsort(-y_score)
    y_true_sorted = np.array(y_true)[order][:k]

    ideal_order = np.argsort(-np.array(y_true))
    ideal_sorted = np.array(y_true)[ideal_order][:k]

    dcg_val = dcg(y_true_sorted)
    idcg_val = dcg(ideal_sorted)

    return dcg_val / (idcg_val + 1e-9)
