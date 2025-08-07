from collections import Counter

import numpy as np
from scipy.stats import bernoulli

from ..optimized_qbs.qbs import Diffix, DPLaplace, SimpleQBS, TableBuilder


def get_indexes_unique(dataset, skip_last_col=False):
    num_attributes = dataset.shape[1]
    # Unique in pseudo-IDs: exclude the last (sensitive) column.
    _, idxs, counts = np.unique(
        dataset[:, : num_attributes - int(skip_last_col)],
        axis=0,
        return_index=True,
        return_counts=True,
    )
    unique_idxs = sorted(list(idxs[counts == 1]))
    return unique_idxs


def init_qbs(dataset, qbs_type, threshold, noise_scale, epsilon, seed):
    # The QBS expects this format.
    tupled_dataset = [tuple(x) for x in dataset]
    if qbs_type == "diffix":
        qbs = Diffix(tupled_dataset, seed=seed)
    elif qbs_type == "simple":
        qbs = SimpleQBS(
            tupled_dataset,
            bucket_threshold=threshold,
            noise_scale=noise_scale,
            seed=seed,
        )
    elif qbs_type == "table-builder":
        qbs = TableBuilder(tupled_dataset, seed=seed)
    elif qbs_type == "dp-laplace":
        qbs = DPLaplace(tupled_dataset, epsilon, seed=seed)
    else:
        raise ValueError("Invalid value for the `qbs_type` parameter.")
    return qbs


def add_occurrences_to_list(l):
    """
    Given a list `l` of objects (that can be hashed), returns the list of
    pairs (object, i), where i ranges from 0 to occurrences(object) - 1 and
    object ranges over the unique elements of `l`.

    For instance if l = [1, 2, 1, 3], the unique elements are {1, 2, 3},
    therefore the method returns [(1, 0), (1, 1), (2, 0), (3, 0)].
    """
    obj_to_count = Counter(l)
    pairs = []
    for obj, count in obj_to_count.items():
        pairs += [(obj, i) for i in range(count)]
    return pairs


def list_to_pdf(l):
    """
    Given a list of values returns the list of unique values together with the
    fraction of values equal to that value in the original list.
    """
    obj_to_count = Counter(l)
    pairs = [(obj, count / len(l)) for obj, count in obj_to_count.items()]
    return pairs


def add_randomized_sensitive_attribute(dataset):
    """
    Returns the dataset with an extra column replacing the sensitive
    attribute. The column is binary and randomized.
    """
    sensitive_attribute = bernoulli.rvs(p=0.5, size=len(dataset))
    return np.hstack((dataset, sensitive_attribute.reshape(-1, 1)))
