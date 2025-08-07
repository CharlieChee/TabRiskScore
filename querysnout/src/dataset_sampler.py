import copy
from collections import defaultdict

import numpy as np
from src.helpers.utils import add_randomized_sensitive_attribute


def init_dataset_sampler(
    dataset_sampler,
    auxiliary_split,
    test_split,
    target_user,
    target_dataset_size,
    target_dataset_seed,
):
    assert dataset_sampler in [
        "without_replacement",
        "exact",
    ], f"ERROR: Invalid dataset sampler {dataset_sampler}"
    target_record = test_split[target_user]
    print("target record shape", target_record)
    if dataset_sampler == "without_replacement":
        # The shadow dataset records are sampled without replacement from the
        # auxiliary dataset.
        return AuxiliaryWithoutReplacementSampler(
            target_record, auxiliary_split, target_dataset_size
        )

    if dataset_sampler == "exact":
        # Sample a target dataset from the test split.
        target_dataset_sampler = TargetDatasetSampler(
            test_split, target_record, target_dataset_size
        )
        target_dataset, target_user = target_dataset_sampler.sample_dataset(
            target_dataset_seed
        )
        # The shadow datasets are the same as the target dataset, but for the
        # sensitive attribute(s) of the target user(s), which are randomized.
        return SameTargetButRandomSensitiveAttributeSampler(target_dataset, target_user)


class DatasetSampler(object):

    def sample_dataset(self):
        raise NotImplementedError


class TargetDatasetSampler(DatasetSampler):
    """
    Generates target datasets having records sampled without replacement from
    a given (larger) dataset, such that the target record is unique in the
    dataset.

    The sampling method is seeded, so that the same dataset can be used
    across different attacks.
    """

    def __init__(self, test_split, target_record, dataset_size):
        self.test_split = test_split
        self.target_record = tuple(target_record)
        self.dataset_size = dataset_size

        # Remove from the test split all the records that are identical to the
        # target record.
        # target_record = tuple(self.target_record)
        # print('target_record', target_record)
        test_split = [
            record for record in self.test_split if tuple(record) != self.target_record
        ]
        # print(f'Removed {len(self.test_split) - len(test_split)} records')
        self.test_split = np.array(test_split)

    def sample_dataset(self, seed):
        np.random.seed(seed)
        dataset_size = min(len(self.test_split), self.dataset_size - 1)
        indexes = np.random.choice(
            len(self.test_split), size=dataset_size, replace=False
        )

        other_records = self.test_split[indexes]

        # We add the target record at a random position in the dataset.
        idx_target = np.random.choice(dataset_size + 1)

        dataset = np.concatenate(
            (
                other_records[:idx_target],
                [self.target_record],
                other_records[idx_target:],
            ),
            axis=0,
        )

        # Add a randomized column (the sensitive attribute).
        dataset = add_randomized_sensitive_attribute(dataset)
        return dataset, idx_target


class AuxiliaryWithoutReplacementSampler(DatasetSampler):
    """
    Generates shadow datasets having records sampled without replacement from
    a given auxiliary dataset. The auxiliary dataset is first partitioned into
    a train and a validation split.
    """

    def __init__(self, target_record, auxiliary_dataset, dataset_size):
        assert dataset_size < len(
            auxiliary_dataset
        ), f"ERROR: Cannot sample without replacement {dataset_size}/{len(auxiliary_dataset)} records."
        self.dataset_size = dataset_size
        self.target_record = tuple(target_record)
        self.name = "auxiliary"

        # Remove from the auxiliary dataset all the records that are
        # identical to the target record.
        # print('target_record', target_record)
        auxiliary_dataset_without_target = [
            aux_record
            for aux_record in auxiliary_dataset
            if tuple(aux_record) != self.target_record
        ]
        # print(f'Removed {len(auxiliary_dataset) - len(auxiliary_dataset_without_target)} records')
        self.split = dict()
        self.split["train"] = np.array(
            auxiliary_dataset_without_target[
                : len(auxiliary_dataset_without_target) // 2
            ]
        )
        self.split["eval"] = np.array(
            auxiliary_dataset_without_target[
                len(auxiliary_dataset_without_target) // 2 :
            ]
        )
        # print(self.split)

    def sample_dataset(self, eval_type):
        dataset_size = min(len(self.split[eval_type]), self.dataset_size - 1)
        indexes = np.random.choice(
            len(self.split[eval_type]), size=dataset_size, replace=False
        )

        other_records = self.split[eval_type][indexes]

        idx_target = np.random.choice(dataset_size + 1)

        dataset = np.concatenate(
            (
                other_records[:idx_target],
                [self.target_record],
                other_records[idx_target:],
            ),
            axis=0,
        )

        # Add a randomized column (the sensitive attribute).
        dataset = add_randomized_sensitive_attribute(dataset)
        return dataset, [idx_target]


class SameTargetButRandomSensitiveAttributeSampler(DatasetSampler):
    """
    Generates shadow datasets identical to the target dataset, but for the
    sensitive attribute value(s) of the target user(s), which is (are)
    randomized.
    """

    def __init__(self, target_dataset, target_user):
        self.target_dataset = target_dataset
        self.target_user = target_user
        self.name = "exact"

    def sample_dataset(self):
        dataset = np.copy(self.target_dataset)
        # Assign a random value to the sensitive attribute value of the
        # target user.
        value = np.random.randint(2)
        dataset[self.target_user, -1] = value

        return dataset, [self.target_user]
