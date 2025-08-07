import sys

sys.path.append("../")

### add feature extractors
import concurrent.futures
import itertools
import multiprocessing as mp
from copy import deepcopy

import numpy as np
import pandas as pd
from optimqbs import qbs
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

######### Concurrent functions #########


# -------------------------------
# ❶ 顶层：apply_feature_extractor_to_datasets
# -------------------------------
def apply_feature_extractor_to_datasets(
    datasets_train: list,
    datasets_eval: list,
    target_record: pd.DataFrame,
    ohe: OneHotEncoder,
    ohe_columns: list,
    ohe_column_names: list,
    continuous_cols: list,
    feature_extractors: list,
    do_ohe: list,
):
    """
    Apply feature extraction to both training and evaluation datasets.
    （文档字符串保持原样，略）
    """

    queries_list_train = [None] * len(feature_extractors)
    queries_list_eval = [None] * len(feature_extractors)

    synth_datasets_train = [d[0] for d in datasets_train]
    membership_labels_train = [d[1] for d in datasets_train]

    synth_datasets_eval = [d[0] for d in datasets_eval]
    membership_labels_eval = [d[1] for d in datasets_eval]

    # ——构造查询（保持原实现）——————————
    QUERY_FEATURE_EXTRACTORS = [
        (
            "query",
            range(1, synth_datasets_train[0].shape[1] + 1),
            1e6,
            {"categorical": (1,), "continuous": (3,)},
        )
    ]
    feature_extractors, do_ohe = get_feature_extractors(QUERY_FEATURE_EXTRACTORS)

    queries_list_train, query_extractor_train = create_queries(
        queries_list=queries_list_train,
        feature_extractors=feature_extractors,
        dataset=synth_datasets_train[0],
        ohe_columns=ohe_columns,
        continuous_cols=continuous_cols,
    )
    queries_list_eval, query_extractor_eval = create_queries(
        queries_list=queries_list_eval,
        feature_extractors=feature_extractors,
        dataset=synth_datasets_eval[0],
        ohe_columns=ohe_columns,
        continuous_cols=continuous_cols,
    )

    # ——改成顺序循环——————————
    features_and_labels = []
    for i in range(len(datasets_train)):
        # 训练集
        features_and_labels.append(
            apply_feature_extractor_one_dataset_parallel(
                dataset=synth_datasets_train[i],
                target_record=target_record,
                ohe=ohe,
                ohe_columns=ohe_columns,
                ohe_column_names=ohe_column_names,
                continuous_cols=continuous_cols,
                feature_extractors=feature_extractors,
                do_ohe=do_ohe,
                queries_list=queries_list_train,
                query_extractor=query_extractor_train,
                train=True,
                membership_label=membership_labels_train[i],
                i=i,
            )
        )
        # 评估集
        features_and_labels.append(
            apply_feature_extractor_one_dataset_parallel(
                dataset=synth_datasets_eval[i],
                target_record=target_record,
                ohe=ohe,
                ohe_columns=ohe_columns,
                ohe_column_names=ohe_column_names,
                continuous_cols=continuous_cols,
                feature_extractors=feature_extractors,
                do_ohe=do_ohe,
                queries_list=queries_list_eval,
                query_extractor=query_extractor_eval,
                train=False,
                membership_label=membership_labels_eval[i],
                i=i,
            )
        )

    print("apply_feature_extractor_to_datasets finished")
    return features_and_labels


# -------------------------------
# ❷ 内层：apply_feature_extractor_one_dataset_parallel
# -------------------------------
def apply_feature_extractor_one_dataset_parallel(
    dataset: pd.DataFrame,
    target_record: pd.DataFrame,
    ohe: OneHotEncoder,
    ohe_columns: list,
    ohe_column_names: list,
    continuous_cols: list,
    feature_extractors: list,
    do_ohe: list,
    queries_list: list,
    query_extractor,
    train: bool,
    membership_label: bool,
    i: int,
) -> tuple:
    """
    Apply feature extraction for a single dataset.
    （文档字符串保持原样，略）
    """

    # ——预处理 OHE（保持原实现）——————————
    if sum(do_ohe) != 0:
        data_ohe = apply_ohe(
            dataset, ohe, ohe_columns, ohe_column_names, continuous_cols
        )
        target_ohe = apply_ohe(
            target_record, ohe, ohe_columns, ohe_column_names, continuous_cols
        )
    else:
        data_ohe, target_ohe = None, None

    # ——顺序提取全部特征——————————
    features_and_column_names = []
    for idx in range(len(feature_extractors)):
        features_and_column_names.append(
            extract_one_feature(
                feature_extractor=feature_extractors[idx],
                queries=queries_list[idx],
                dataset=dataset,
                ohe_columns=ohe_columns,
                target_record=target_record,
                query_extractor=query_extractor,
                do_ohe=do_ohe[idx],
                data_ohe=data_ohe,
                ohe_column_names=ohe_column_names,
                continuous_cols=continuous_cols,
                target_ohe=target_ohe,
            )
        )

    # ——拼接结果——————————
    all_feature_one_ds = [f[0] for f in features_and_column_names]
    all_feature_names = [f[1] for f in features_and_column_names]

    X = pd.DataFrame(np.array(all_feature_one_ds), columns=all_feature_names)
    return X, membership_label, train


def extract_one_feature(
    feature_extractor,
    queries,
    dataset,
    ohe_columns,
    target_record,
    query_extractor,
    do_ohe,
    data_ohe,
    ohe_column_names,
    continuous_cols,
    target_ohe,
):
    """
    Extract features using a given feature extractor.

    :param feature_extractor: The feature extractor function or a tuple containing the function and additional parameters.
    :type feature_extractor: function or tuple
    :param queries: A list of queries for extracting features.
    :type queries: list
    :param dataset: The dataset containing the features for extraction.
    :type dataset: pd.DataFrame
    :param ohe_columns: A list of column names representing one-hot encoded categorical features.
    :type ohe_columns: list
    :param target_record: The target record for which features are to be extracted.
    :type target_record: pd.DataFrame
    :param query_extractor: The function used for extracting features when the feature extractor is a tuple.
    :type query_extractor: function
    :param do_ohe: A boolean indicating whether one-hot encoding is required.
    :type do_ohe: bool
    :param data_ohe: The one-hot encoded version of the dataset.
    :type data_ohe: pd.DataFrame
    :param ohe_column_names: The names of the columns of the one-hot encoding result.
    :type ohe_column_names: list
    :param continuous_cols: A list of column names representing continuous features.
    :type continuous_cols: list
    :param target_ohe: The one-hot encoded version of the target record.
    :type target_ohe: pd.DataFrame

    :returns: A tuple containing:
        - features (list): The extracted features.
        - col_names (list): The names of the extracted features.
    :rtype: tuple
    """
    if isinstance(feature_extractor, tuple):
        dataset_int = dataset.copy()
        dataset_int[ohe_columns] = dataset[ohe_columns].astype(int)
        target_record_int = target_record.copy()
        target_record_int[ohe_columns] = target_record_int[ohe_columns].astype(int)
        features, col_names = query_extractor(dataset_int, target_record_int, queries)
    elif do_ohe:
        features, col_names = feature_extractor(
            data_ohe,
            ohe_columns,
            ohe_column_names,
            continuous_cols,
            target_ohe,
        )
    else:
        features, col_names = feature_extractor(
            dataset,
            ohe_columns,
            ohe_column_names,
            continuous_cols,
            target_record,
        )
    return features, col_names


######### Utility functions and feature extractors #########


def fit_ohe(df: pd.DataFrame, categorical_cols: list, metadata: dict) -> tuple:
    # first extract all categories from the metadata
    meta_data_columns = [col["name"] for col in metadata]
    categories = []
    for col in categorical_cols:
        categories.append(metadata[meta_data_columns.index(col)]["representation"])

    ohe = OneHotEncoder(categories=categories)
    ohe.fit(df[categorical_cols])

    ohe_column_names = []
    all_categories = ohe.categories_
    for i, categories in enumerate(all_categories):
        for category in categories:
            ohe_column_names.append(categorical_cols[i] + "_" + str(category))

    return ohe, ohe_column_names


def apply_ohe(
    df: pd.DataFrame,
    ohe: OneHotEncoder,
    categorical_cols: list,
    ohe_column_names: list,
    continous_cols: list,
) -> pd.DataFrame:
    """One-hot-encode dataset

    :param df: dataset to encode
    :type df: pd.DataFrame
    :param ohe: OneHotEncoder instance
    :type ohe: OneHotEncoder
    :param categorical_cols: names of categorical columns
    :type categorical_cols: list
    :param ohe_column_names: names for one-hot-encoded columns
    :type ohe_column_names: list
    :param continous_cols: names of continuous columns
    :type continous_cols: list
    :return: One-hot-encoded dataset
    :rtype: pd.DataFrame
    """
    ohe_values = ohe.transform(df[categorical_cols]).toarray()
    ohe_df = pd.DataFrame(data=ohe_values, columns=ohe_column_names, index=df.index)
    results_df = df[continous_cols].merge(ohe_df, left_index=True, right_index=True)

    return results_df


def extract_naive_features(
    synthetic_df: pd.DataFrame,
    categorical_cols: list,
    ohe_column_names: list,
    continuous_cols: list,
    target_record=pd.DataFrame,
) -> tuple:
    """Compute the Naive method as described in "Synthetic data -- anonymisation groundhog day" (Usenix 2022)

    :param synthetic_df: Synthetic dataset
    :type synthetic_df: pd.DataFrame
    :param categorical_cols: names of categorical columns
    :type categorical_cols: list
    :param ohe_column_names: names of one-hot encoded columns
    :type ohe_column_names: list
    :param continuous_cols: names of continuous columns
    :type continuous_cols: list
    :param target_record: dataframe containing target record, defaults to pd.DataFrame
    :type target_record: pd.DataFrame, optional
    :return: extracted features and names
    :rtype: tuple
    """
    ## (1) For each continuous col, extract the mean, median and variance
    # get mean, median and var for each col
    means = [np.mean(synthetic_df[col]) for col in continuous_cols]
    medians = [np.median(synthetic_df[col]) for col in continuous_cols]
    varians = [np.var(synthetic_df[col]) for col in continuous_cols]
    features = means + medians + varians
    # get col names
    col_names = ["mean_" + col for col in continuous_cols]
    col_names += ["median_" + col for col in continuous_cols]
    col_names += ["var_" + col for col in continuous_cols]

    ## (2) For each categorical col, extract the the number of distinct categories plus the most and least frequent category
    for cat_col in categorical_cols:
        all_ohe_cols = [i for i in ohe_column_names if i.split("_")[0] == cat_col]
        all_summed = synthetic_df[all_ohe_cols].sum()
        distinct = sum(all_summed > 0)
        most_freq = int(all_summed.index[np.argmax(all_summed.values)].split("_")[1])
        least_freq = int(all_summed.index[np.argmin(all_summed.values)].split("_")[1])
        features += [distinct, most_freq, least_freq]
        col_names += [
            f"{cat_col}_distinct",
            f"{cat_col}_most_freq",
            f"{cat_col}_least_freq",
        ]

    return features, col_names


def extract_correlation_features(
    synthetic_df: pd.DataFrame,
    categorical_cols: list,
    ohe_column_names: list,
    continuous_cols: list,
    target_record=pd.DataFrame,
) -> tuple:
    """Extract correlation features

    :param synthetic_df: Synthetic dataset
    :type synthetic_df: pd.DataFrame
    :param categorical_cols: names of categorical columns
    :type categorical_cols: list
    :param ohe_column_names: names of one-hot encoded columns
    :type ohe_column_names: list
    :param continuous_cols: names of continuous columns
    :type continuous_cols: list
    :param target_record: dataframe containing target record, defaults to pd.DataFrame
    :type target_record: pd.DataFrame, optional
    :return: extracted features and names
    :rtype: tuple
    """
    corr_matrix = synthetic_df.corr()
    # replace nan values with 0
    corr_matrix = corr_matrix.fillna(0.0)
    # Remove redundant entries from the symmetrical matrix.
    above_diagonal = np.triu_indices(corr_matrix.shape[0], 1)
    features = list(corr_matrix.values[above_diagonal])

    # get col names
    col_names = ["corr_" + str(i) for i in range(len(features))]

    return features, col_names


def get_queries(
    orders,
    categorical_indices: list,
    continous_indices: list,
    num_cols: int,
    number: int,
    cat_condition_options: tuple = (-1, 1),
    cont_condition_options: tuple = (3, -3),
    random_state: int = 42,
) -> list:
    """
    Condition options:
         0  ->  no condition on this attribute;
         1  ->  ==
        -1  ->  !=
         2  ->  >
         3  ->  >=
        -2  ->  <
        -3  ->  <=

    :param orders: list containing numbers of features to consider when extracting queries from feature combinations
    :type orders: list
    :param categorical_indices: indices of categorical columns
    :type categorical_indices: list
    :param continous_indices: indices of continuous columns
    :type continous_indices: list
    :param num_cols: number of columns in dataset
    :type num_cols: int
    :param number: number of combinations to return
    :type number: int
    :param cat_condition_options: defaults to (-1, 1)
    :type cat_condition_options: tuple, optional
    :param cont_condition_options:  defaults to (3, -3)
    :type cont_condition_options: tuple, optional
    :param random_state: random seed, defaults to 42
    :type random_state: int, optional
    :return: extracted features and names
    :rtype: list
    """

    all_combinations = []

    for order in orders:
        all_indices = list(itertools.combinations(range(num_cols), order))
        for indices in all_indices:
            indices_combinations = []
            for i, index in enumerate(indices):
                if index in categorical_indices:
                    index_options = cat_condition_options
                else:
                    index_options = cont_condition_options
                if i == 0:
                    for index_option in index_options:
                        base_tup = np.array([0] * num_cols)
                        base_tup[index] = index_option
                        indices_combinations.append(base_tup)
                else:
                    for j, index_option in enumerate(index_options):
                        if j == 0:
                            for base_tup in indices_combinations:
                                base_tup[index] = index_option
                        else:
                            indices_combinations_c = deepcopy(indices_combinations)
                            for base_tup in indices_combinations_c:
                                base_tup[index] = index_option
                            indices_combinations += indices_combinations_c
            for combo in indices_combinations:
                all_combinations.append(tuple(combo))

    if number < len(all_combinations):
        np.random.seed(random_state)
        indices = np.random.choice(len(all_combinations), replace=False, size=(number,))
        queries = [all_combinations[idx] for idx in indices]
    else:
        queries = all_combinations

    return queries


def feature_extractor_queries_CQBS(
    synthetic_df: pd.DataFrame, target_record: pd.DataFrame, queries: list
):
    """Extract query-based features

    :param synthetic_df: synthetic data
    :type synthetic_df: pd.DataFrame
    :param target_record: target record
    :type target_record: pd.DataFrame
    :param queries: queries
    :type queries: list
    :return: extracted features and names
    :rtype: tuple
    """
    # set up qbs of synthetic dataframe and define target values
    # qbs_data = qbs.SimpleQBS(synthetic_df.itertuples(index=False, name=None))

    try:
        qbs_data = qbs.SimpleQBS(synthetic_df.itertuples(index=False, name=None))
    except Exception:
        import sys
        import traceback

        traceback.print_exc()
        sys.exit(1)
    target_values = [tuple(target_record.values[0])]

    # get features by batch-quering using the queries and qbs
    features = qbs_data.query(target_values * len(queries), queries)

    # get feature names
    og_data_columns = synthetic_df.columns
    col_names = [
        "_".join(
            [
                f"{cond}_{og_data_columns[i]}"
                for i, cond in enumerate(conditions)
                if cond != 0
            ]
        )
        for conditions in queries
    ]

    return features, col_names


def feature_extractor_topX_full(
    synthetic_df: pd.DataFrame, target_record_ohe: pd.DataFrame, top_X: int = 50
):
    """Extract features based on top X cosine similarity between synthetic data and target record

    :param synthetic_df: synthetic data
    :type synthetic_df: pd.DataFrame
    :param target_record_ohe: one-hot encoded target record
    :type target_record_ohe: pd.DataFrame
    :param top_X: number of most similar records to consider, defaults to 50
    :type top_X: int, optional
    :return: extracted features and names
    :rtype: tuple
    """
    all_cos_sim = np.array(
        [
            cosine_similarity(
                synthetic_df.iloc[i].values.reshape(1, -1),
                target_record_ohe.values.reshape(1, -1),
            )[0][0]
            for i in range(len(synthetic_df))
        ]
    )
    ordered_idx = np.argsort(all_cos_sim)[::-1]

    top_x_data = synthetic_df.iloc[ordered_idx[:top_X]]

    features = list(top_x_data.values.flatten())
    col_names = []

    for i in range(top_X):
        col_names += [k + "_top_X=" + str(i) for k in top_x_data.columns]

    return features, col_names


def feature_extractor_distances(
    synthetic_df: pd.DataFrame, target_record_ohe: pd.DataFrame
):
    """Extract features based on cosine similarity between synthetic data and target record

    :param synthetic_df: synthetic data
    :type synthetic_df: pd.DataFrame
    :param target_record_ohe: one-hot encoded target record
    :type target_record_ohe: pd.DataFrame
    :return: extracted features and names
    :rtype: tuple
    """
    all_cos_sim = np.array(
        [
            cosine_similarity(
                synthetic_df.iloc[i].values.reshape(1, -1),
                target_record_ohe.values.reshape(1, -1),
            )[0][0]
            for i in range(len(synthetic_df))
        ]
    )
    ordered_vals = np.sort(all_cos_sim)[::-1]

    features = list(ordered_vals)
    col_names = ["distance_X=" + str(k) for k in range(len(features))]

    return features, col_names


def apply_feature_extractor_sequential(
    datasets: list,
    target_record: pd.DataFrame,
    labels: list,
    ohe: OneHotEncoder,
    ohe_columns: list,
    ohe_column_names: list,
    continuous_cols: list,
    feature_extractors: list,
    do_ohe: list,
) -> tuple:
    """
    Given a list of feature extractor functions and synthetic datasets, extract all features and
    create a new DataFrame with all features per dataset as individual records.

    :param datasets: A list of shadow synthetic datasets.
    :type datasets: list
    :param target_record: DataFrame of one record with the target record, potentially to be used by the feature extractor.
    :type target_record: pd.DataFrame
    :param labels: A list of labels corresponding to the datasets.
    :type labels: list
    :param ohe: A fitted one-hot encoder instance.
    :type ohe: OneHotEncoder
    :param ohe_columns: The columns on which the one-hot encoding should be applied.
    :type ohe_columns: list
    :param ohe_column_names: The names of the columns of the one-hot encoding result.
    :type ohe_column_names: list
    :param continuous_cols: The columns that are continuous.
    :type continuous_cols: list
    :param feature_extractors: A list of feature extractor functions. All functions have as input a dataset and output a list of features and a list of column names. If more than one feature extractor is specified, all features are extracted and appended.
    :type feature_extractors: list
    :param do_ohe: A list of boolean values indicating whether each feature extractor function requires the dataset to be one-hot encoded or not.
    :type do_ohe: list

    :returns: DataFrame containing all features per dataset and the corresponding labels.
    :rtype: pd.DataFrame
    """
    all_features = []

    # for k, dataset in tqdm(enumerate(datasets)):
    for k, dataset in enumerate(datasets):
        if sum(do_ohe) != 0:
            data_ohe = apply_ohe(
                dataset, ohe, ohe_columns, ohe_column_names, continuous_cols
            )
            target_ohe = apply_ohe(
                target_record,
                ohe,
                ohe_columns,
                ohe_column_names,
                continuous_cols,
            )
        all_feature_one_ds = []
        all_feature_names = []
        for i, feature_extractor in enumerate(feature_extractors):
            if isinstance(feature_extractor, tuple):
                # then we know it's query extracting with additional params
                query_extractor, orders, number, conditions = feature_extractor

                # make sure to compute the queries only once
                if k == 0:
                    all_columns = list(dataset.columns)
                    categorical_indices = [
                        all_columns.index(col) for col in ohe_columns
                    ]
                    continous_indices = [
                        all_columns.index(col) for col in continuous_cols
                    ]
                    queries = get_queries(
                        orders=orders,
                        categorical_indices=categorical_indices,
                        continous_indices=continous_indices,
                        num_cols=dataset.shape[1],
                        number=number,
                        cat_condition_options=conditions["categorical"],
                        cont_condition_options=conditions["continuous"],
                    )
                # for C QBS we need int for categorical
                dataset_int = dataset.copy()
                dataset_int[ohe_columns] = dataset[ohe_columns].astype(int)
                target_record_int = target_record.copy()
                target_record_int[ohe_columns] = target_record_int[ohe_columns].astype(
                    int
                )
                features, col_names = query_extractor(
                    dataset_int, target_record_int, queries
                )
            elif do_ohe[i]:
                features, col_names = feature_extractor(
                    data_ohe,
                    ohe_columns,
                    ohe_column_names,
                    continuous_cols,
                    target_ohe,
                )
            else:
                features, col_names = feature_extractor(
                    dataset,
                    ohe_columns,
                    ohe_column_names,
                    continuous_cols,
                    target_record,
                )
            all_feature_one_ds += features
            all_feature_names += col_names
        all_features.append(all_feature_one_ds)

    shadow_train_X = pd.DataFrame(
        data=np.array(all_features), columns=all_feature_names
    )

    return shadow_train_X, labels


def create_queries(
    queries_list: list,
    feature_extractors: list,
    dataset: pd.DataFrame,
    ohe_columns: list,
    continuous_cols: list,
):
    """
    Generate queries based on the provided feature extractors and dataset.

    :param queries_list: A list to store the generated queries for each feature extractor.
    :type queries_list: list
    :param feature_extractors: A list of feature extractors, where each element can be a function or a tuple with parameters.
    :type feature_extractors: list
    :param dataset: The dataset containing the features for query generation.
    :type dataset: pd.DataFrame
    :param ohe_columns: A list of column names representing one-hot encoded categorical features.
    :type ohe_columns: list
    :param continuous_cols: A list of column names representing continuous features.
    :type continuous_cols: list

    :returns: A tuple containing:
        - queries_list (list): The updated list of queries generated for each feature extractor.
        - query_extractor: The last used query extractor from the feature extractors list.
    :rtype: tuple
    """
    for i, feature_extractor in enumerate(feature_extractors):
        query_extractor, orders, number, conditions = feature_extractor
        all_columns = list(dataset.columns)
        categorical_indices = [all_columns.index(col) for col in ohe_columns]
        continous_indices = [all_columns.index(col) for col in continuous_cols]
        queries = get_queries(
            orders=orders,
            categorical_indices=categorical_indices,
            continous_indices=continous_indices,
            num_cols=dataset.shape[1],
            number=number,
            cat_condition_options=conditions["categorical"],
            cont_condition_options=conditions["continuous"],
        )
        queries_list[i] = queries
    return queries_list, query_extractor


def get_feature_extractors(feature_extractor_names: list) -> tuple:
    """
    Given a list of strings or tuples specifying the feature extractors to be used,
    create a list of the corresponding functions and parameters.

    :param feature_extractor_names: A list of feature extractors, where each element can be:
        - A string specifying a feature extractor (e.g., 'naive', 'correlation', 'closest_X_full', 'all_distances')
        - A tuple with a feature extractor name and additional parameters (name, orders, number, conditions)
    :type feature_extractor_names: list

    :returns: A tuple containing:
        - feature_extractors (list): A list of functions (or tuples with functions and parameters) corresponding to the requested feature extractors.
        - do_ohe (list): A list of boolean values indicating whether one-hot encoding (OHE) should be performed for each feature extractor.
    :rtype: tuple
    """
    feature_extractors, do_ohe = [], []
    for feat in feature_extractor_names:
        if isinstance(feat, str):
            if feat == "naive":
                feature_extractors.append(extract_naive_features)
                do_ohe.append(True)
            elif feat == "correlation":
                feature_extractors.append(extract_correlation_features)
                do_ohe.append(True)
            elif feat == "closest_X_full":
                feature_extractors.append(feature_extractor_topX_full)
                do_ohe.append(True)
            elif feat == "all_distances":
                feature_extractors.append(feature_extractor_distances)
                do_ohe.append(True)
            else:
                print("Not a valid feature extractor")
        elif isinstance(feat, tuple):
            name, orders, number, conditions = feat
            if name == "query":
                feature_extractors.append(
                    (feature_extractor_queries_CQBS, orders, number, conditions)
                )
                do_ohe.append(False)
            else:
                print("Not a valid feature extractor")
        else:
            print("Not a valid feature extractor")

    return feature_extractors, do_ohe
