### Add pipeline to create data for shadow modeling
import concurrent.futures
import pickle as pickle
from random import sample

import pandas as pd

from .generators import get_generator
from .utils import blockPrint, enablePrint

### Parallelized functions for generating shadow and evaluation synthetic datasets


def generate_dataset_parallel(
    df_aux: pd.DataFrame,
    df_target: pd.DataFrame,
    meta_data: list,
    target_record: pd.DataFrame,
    df_eval: pd.DataFrame,
    in_dataset: bool,
    generator_name: str,
    n_synth: int,
    n_original: int,
    seeds_train: list,
    seeds_eval: list,
    idx: int,
    shadow_datasets: list,
    shadow_membership_labels: list,
    evaluation_datasets: list,
    evaluation_membership_labels: list,
    epsilon: float,
    train: bool,
):
    """
    This function trains one generator and generates a single dataset. The generator is trained on `df_target`,
    either with the target record swapped out for a different record sampled from `df_eval` (in_dataset=False),
    or on the full `df_target` containing the target record (in_dataset=True). The generated synthetic dataset and
    its membership label are placed in the corresponding lists.

    :param df_target: Target dataset.
    :type df_target: pandas.DataFrame
    :param meta_data: List containing metadata concerning the data (feature types, ranges, etc.), necessary for training synthetic data generators.
    :type meta_data: list
    :param target_record: DataFrame containing only the target record.
    :type target_record: pandas.DataFrame
    :param df_eval: Evaluation pool to draw the reference record from.
    :type df_eval: pandas.DataFrame
    :param in_dataset: If True, the target record is in the dataset used to train the synthetic data generator.
        If False, it is replaced by a reference record randomly sampled from `df_eval`.
    :type in_dataset: bool
    :param generator_name: Name of the generator used, e.g., 'SYNTHPOP', 'BAYNET', etc. See reprosyn library for more details.
    :type generator_name: str
    :param n_synth: Size of each synthetic dataset.
    :type n_synth: int
    :param seeds: List of seed values; length must be equal to `n_datasets`.
    :type seeds: list
    :param idx: Counter for number of generated synthetic datasets.
    :type idx: int
    :param synthetic_datasets: List containing all generated synthetic datasets. This function generates a single synthetic dataset and places it in the list, selecting the position according to `idx`.
    :type synthetic_datasets: list
    :param membership_labels: List containing membership labels for each synthetic dataset. If the target record is included in the training set, the membership label is 1, otherwise it is 0. Label at index `idx` in `membership_labels` corresponds to the synthetic dataset at index `idx` in `synthetic_datasets`.
    :type membership_labels: list
    :param epsilon: Epsilon when training with differential privacy.
    :type epsilon: float

    :return: synthetic dataset,
             bool indicating whether the synthetic data generator was trained on the target record,
             bool indicating whether the synthetic dataset is used for MIA training or evaluation
    :rtype: tuple
    """
    generator = get_generator(generator_name, epsilon=epsilon)
    if train:
        if in_dataset:
            indices = sample(list(df_aux.index), n_original - 1)
            df_train = pd.concat([df_aux.loc[indices], target_record], axis=0)
        else:
            indices = sample(list(df_aux.index), n_original)
            df_train = df_aux.loc[indices]

        blockPrint()
        synthetic_dataset = generator.fit_generate(
            dataset=df_train,
            metadata=meta_data,
            size=n_synth,
            seed=seeds_train[idx],
        )
        enablePrint()
    else:
        if in_dataset:
            df_train = pd.concat([df_target, target_record], axis=0)
        else:
            reference_record = df_eval.sample(1)
            df_train = pd.concat([df_target, reference_record], axis=0)
        blockPrint()
        synthetic_dataset = generator.fit_generate(
            dataset=df_train,
            metadata=meta_data,
            size=n_synth,
            seed=seeds_eval[idx],
        )
        enablePrint()

    if train:
        shadow_datasets[idx] = synthetic_dataset
        shadow_membership_labels[idx] = in_dataset
    else:
        evaluation_datasets[idx] = synthetic_dataset
        evaluation_membership_labels[idx] = in_dataset

    return synthetic_dataset, in_dataset, train


def generate_datasets_parallel(
    df_aux: pd.DataFrame,
    df_target: pd.DataFrame,
    meta_data: list,
    target_record: pd.DataFrame,
    df_eval: pd.DataFrame,
    generator_name: str,
    n_synth: int,
    n_original: int,
    n_datasets: int,
    seeds_train: list,
    seeds_eval: list,
    epsilon: float,
    train: bool,
):
    """
    This function allows evaluation datasets to be generated concurrently. It is not meant to be called directly,
    but rather from `generate_evaluation_datasets`. Each launched task corresponds to a single synthetic data generator being trained
    and used to generate a single synthetic dataset. Exactly half of the synthetic data generators are trained on data including the target record.

    :param df_target: Target dataset.
    :type df_target: pandas.DataFrame
    :param meta_data: List containing metadata concerning the data (feature types, ranges, etc.), necessary for training synthetic data generators.
    :type meta_data: list
    :param target_record: DataFrame containing only the target record.
    :type target_record: pandas.DataFrame
    :param df_eval: Evaluation pool to draw the reference record from.
    :type df_eval: pandas.DataFrame
    :param generator_name: Name of the generator used, e.g., 'SYNTHPOP', 'BAYNET', etc. See reprosyn library for more details.
    :type generator_name: str
    :param n_synth: Size of each synthetic dataset.
    :type n_synth: int
    :param n_datasets: Number of synthetic datasets to generate.
    :type n_datasets: int
    :param seeds: List of seed values; length must be equal to `n_datasets`.
    :type seeds: list
    :param epsilon: Epsilon when training with differential privacy.
    :type epsilon: float

    :returns: A tuple containing:
        - A list of all generated synthetic datasets.
        - A list of membership labels for each generated synthetic dataset.
    :rtype: tuple
    """
    shadow_datasets = [None] * n_datasets
    shadow_membership_labels = [None] * n_datasets

    evaluation_datasets = [None] * n_datasets
    evaluation_membership_labels = [None] * n_datasets

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(n_datasets * 2):
            in_dataset = i % 2 == 0
            train = i < n_datasets
            if train:
                idx = i
            else:
                idx = i - n_datasets
            futures.append(
                executor.submit(
                    generate_dataset_parallel,
                    df_aux=df_aux,
                    df_target=df_target,
                    meta_data=meta_data,
                    target_record=target_record,
                    df_eval=df_eval,
                    in_dataset=in_dataset,
                    generator_name=generator_name,
                    n_synth=n_synth,
                    n_original=n_original,
                    seeds_train=seeds_train,
                    seeds_eval=seeds_eval,
                    idx=idx,
                    shadow_datasets=shadow_datasets,
                    shadow_membership_labels=shadow_membership_labels,
                    evaluation_datasets=evaluation_datasets,
                    evaluation_membership_labels=evaluation_membership_labels,
                    epsilon=epsilon,
                    train=train,
                )
            )
        datasets_and_labels = [
            f.result() for f in concurrent.futures.as_completed(futures)
        ]
    return datasets_and_labels


def generate_datasets(
    df_aux: pd.DataFrame,
    df_target: pd.DataFrame,
    meta_data: list,
    target_record_id: int,
    df_eval: pd.DataFrame,
    generator_name: str,
    n_synth: int = 1000,
    n_original: int = 1000,
    n_datasets: int = 1000,
    seeds_train: list = None,
    seeds_eval: list = None,
    epsilon: float = 0.0,
):
    """
    Launch the pipeline to generate evaluation synthetic datasets.

    :param df_target: Target dataset.
    :type df_target: pandas.DataFrame
    :param meta_data: List containing metadata concerning the data (feature types, ranges, etc.), necessary for training synthetic data generators.
    :type meta_data: list
    :param target_record_id: Index of the target record.
    :type target_record_id: int
    :param df_eval: Evaluation pool to draw the reference record from.
    :type df_eval: pandas.DataFrame
    :param generator_name: Name of the generator used, e.g., 'SYNTHPOP', 'BAYNET', etc. See reprosyn library for more details.
    :type generator_name: str
    :param n_synth: Size of each synthetic dataset.
    :type n_synth: int
    :param n_datasets: Number of synthetic datasets to generate.
    :type n_datasets: int
    :param seeds: List of seed values; length must be equal to `n_datasets`.
    :type seeds: list
    :param epsilon: Epsilon when training with differential privacy.
    :type epsilon: float

    :returns: A list containing tuples of (synthetic dataset, membership label)
    :rtype: list
    """
    if seeds_train is None:
        seeds_train = list(range(n_datasets))
    if seeds_eval is None:
        seeds_eval = list(range(n_datasets, 2 * n_datasets))

    target_record = df_target.loc[[target_record_id]]

    return generate_datasets_parallel(
        df_aux=df_aux,
        df_target=df_target,
        meta_data=meta_data,
        target_record=target_record,
        df_eval=df_eval,
        generator_name=generator_name,
        n_synth=n_synth,
        n_original=n_original,
        n_datasets=n_datasets,
        seeds_train=seeds_train,
        seeds_eval=seeds_eval,
        epsilon=epsilon,
        train=False,
    )
