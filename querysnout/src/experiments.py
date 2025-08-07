import copy
import multiprocessing
import os
import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.dataset_loader import DatasetLoader
from src.dataset_sampler import TargetDatasetSampler, init_dataset_sampler
from src.helpers.generation_logger import GenerationLogger
from src.helpers.utils import get_indexes_unique, init_qbs
from src.qbs_environment import QBSEnvironment
from src.query_search import (EvolutionaryQuerySearch, MutationProbabilities,
                              RandomQuerySearch)


def search(environment, save_path, args):
    mutation_probs = MutationProbabilities(
        args.p_copy, args.p_modify, args.p_switch, args.p_swap
    )

    if args.search_type == "evolutionary":
        query_search = EvolutionaryQuerySearch(
            environment,
            args.num_attributes + 1,
            args.population_size,
            args.num_queries,
            args.frac_elitism,
            mutation_probs,
            args.model_type,
            args.num_procs,
        )
    else:
        query_search = RandomQuerySearch(
            environment,
            args.num_attributes + 1,
            args.population_size,
            args.num_queries,
            args.model_type,
            args.num_procs,
        )

    print("Initialising the population...")
    query_search.init_population(verbose=False)
    # qs.display_population()
    # print('Fitnesses (train)', qs.fitnesses_train)
    # print('Fitnesses (eval)', qs.fitnesses_eval)

    print("\n[Starting iterations]")
    logger = GenerationLogger(query_search, save_path)

    count_100 = 0
    for g in range(0, args.num_generations + 1):
        if g > 0:
            query_search.generation(verbose=False)
        # Periodical evaluation using an MLP.
        if g % 10 == 0:
            query_search.display_population(1)
        logger.log(g)
        # Early stopping after 10 generations of almost perfect accuracy.
        if logger.max_fitnesses["eval"][-1] >= 0.9999:
            count_100 += 1
            if count_100 == 10:
                print("Early stopping after 10 generations of perfect accuracy")
                break
        else:
            # Resetting the counter
            count_100 = 0
    logger.mark_complete()
    return query_search


def init_environment(dataset_sampler, args, qbs_seeds_to_exclude):
    environment = QBSEnvironment(
        dataset_sampler=dataset_sampler,
        num_datasets=args.num_datasets,
        eval_fraction=args.eval_fraction,
        qbs_type=args.qbs_type,
        qbs_threshold=args.qbs_threshold,
        qbs_noise_scale=args.qbs_noise_scale,
        qbs_epsilon=args.qbs_epsilon,
        verbose=args.verbose,
        num_procs=args.num_procs,
        qbs_seeds_to_exclude=qbs_seeds_to_exclude,
    )
    return environment


def train_model(X_train, y_train, X_eval, y_eval, seed):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    logreg = LogisticRegression(max_iter=1000, random_state=seed)
    logreg = logreg.fit(X_train_scaled, y_train)
    logreg_acc_train = accuracy_score(y_train, logreg.predict(X_train_scaled))
    logreg_acc_eval = accuracy_score(y_eval, logreg.predict(X_eval_scaled))

    mlp = MLPClassifier(
        max_iter=1000,
        hidden_layer_sizes=(50, 20),
        early_stopping=True,
        random_state=seed,
    )
    mlp = mlp.fit(X_train_scaled, y_train)
    mlp_acc_train = accuracy_score(y_train, mlp.predict(X_train_scaled))
    mlp_acc_eval = accuracy_score(y_eval, mlp.predict(X_eval_scaled))

    results = {
        "data": {
            "X_train": X_train,
            "y_train": y_train,
            "X_eval": X_eval,
            "y_eval": y_eval,
            "seed": seed,
            "scaler": scaler,
        },
        "logreg": {
            "model": logreg,
            "acc_train": logreg_acc_train,
            "acc_eval": logreg_acc_eval,
        },
        "mlp": {"model": mlp, "acc_train": mlp_acc_train, "acc_eval": mlp_acc_eval},
    }
    print(
        f"Logreg acc train: {logreg_acc_train:.1%}, acc eval: "
        + f"{logreg_acc_eval:.1%}"
    )
    print(f"MLP acc train: {mlp_acc_train:.1%}, acc eval: {mlp_acc_eval:.1%}")
    return results


def get_prefix(args):
    search_type = "evol" if args.search_type == "evolutionary" else "rand"
    prefix = f"st-{search_type}_na-{args.num_attributes}_"
    # Include the target dataset size in the filename for the scaling
    # experiments only.
    if (
        args.dataset_name in ["adults", "census"] and args.target_dataset_size != 8000
    ) or (args.dataset_name == "insurance" and args.target_dataset_size != 1000):
        prefix += f"tds-{args.target_dataset_size}_"
    prefix += (
        f"nd-{args.num_datasets}_"
        + f"ef-{args.eval_fraction:.2f}_"
        + f"ps-{args.population_size}_"
        + f"nq-{args.num_queries}_"
        + f"ng-{args.num_generations}_"
        + f"pc-{args.p_copy:.3f}_"
        + f"pm-{args.p_modify:.3f}_"
        + f"pswa-{args.p_swap:.2f}_"
        + f"pswi-{args.p_switch:.2f}"
    )
    return prefix


def run_query_search(
    args,
    save_path,
    aux_split,
    test_split,
    tar_user,
    tar_qbs_seeds,
    tar_sensitive_attributes,
    seed,
):
    print(args)
    print("Save path: ", save_path)

    start_time = time.time()
    np.random.seed(seed)

    dataset_sampler = init_dataset_sampler(
        args.dataset_sampler,
        aux_split,
        test_split,
        tar_user,
        args.target_dataset_size,
        target_dataset_seed=tar_user,
    )

    qbs_environment = init_environment(
        dataset_sampler, args, qbs_seeds_to_exclude=tar_qbs_seeds
    )

    query_search = search(qbs_environment, save_path, args)

    # Train a model to predict the sensitive attribute using the queries
    # returned by the query search on data from the QBS environment.
    best_solution = query_search.population[0]
    best_solution_formatted = query_search.format_solution(best_solution)
    if qbs_environment.qbs_type == "dp-laplace":
        best_solution_dp, budgets = zip(*best_solution_formatted)
        best_solution_dp, budgets = list(best_solution_dp), list(budgets)
    elif not qbs_environment.deterministic:
        best_solution = [q[0] for q in best_solution_formatted]
    X_train, X_eval = qbs_environment.get_answers(best_solution_formatted)
    y_train, y_eval = qbs_environment.get_labels()
    print(f"Train/eval set size: {len(y_train)}/{len(y_eval)}")
    results = train_model(X_train, y_train, X_eval, y_eval, seed)
    results["num_queries_shadow_qbs"] = len(qbs_environment._cached_queries)
    print("Number of queries per shadow qbs:", results["num_queries_shadow_qbs"])

    # We test on multiple target QBSes and target datasets.
    X_test, y_test = [], []
    target_dataset_sampler = TargetDatasetSampler(
        test_split, test_split[tar_user], args.target_dataset_size
    )
    for ts in range(args.num_test_samples):
        if args.dataset_sampler == "exact":
            tar_dataset, tar_idx = target_dataset_sampler.sample_dataset(seed=tar_user)
            tar_dataset[tar_idx, -1] = tar_sensitive_attributes[ts]
        else:
            # Sample a target dataset from the target split.
            tar_dataset, tar_idx = target_dataset_sampler.sample_dataset(
                seed=tar_user + ts
            )
        # print(tar_dataset, tar_dataset[tar_idx], tar_idx)
        # We extract the target user's sensitive attribute as the ground truth.
        y_test.append(tar_dataset[tar_idx, -1])
        tar_qbs = init_qbs(
            tar_dataset,
            args.qbs_type,
            args.qbs_threshold,
            args.qbs_noise_scale,
            args.qbs_epsilon,
            tar_qbs_seeds[ts],
        )

        if args.qbs_type == "dp-laplace":
            answers = tar_qbs.structured_query([tar_idx], best_solution_dp, budgets)
            sol_size = len(best_solution_dp)
        else:
            answers = tar_qbs.structured_query([tar_idx], best_solution)
            sol_size = len(best_solution)

        assert (
            type(answers) == list and len(answers) == sol_size
        ), f"ERROR: Invalid number of query answers {len(answers)}: it should be equal to the number of queries."
        X_test.append(answers)
    X_test = np.array(X_test)
    assert len(X_test) == args.num_test_samples
    # print(X_test.shape, len(y_test), np.sum(y_test))

    X_test_scaled = results["data"]["scaler"].transform(X_test)

    logreg_y_probas = results["logreg"]["model"].predict_proba(X_test_scaled)[:, 1]
    logreg_auc_test = get_auc_score(y_test, logreg_y_probas)
    logreg_acc_test = accuracy_score(y_test, logreg_y_probas >= 0.5)

    mlp_y_probas = results["mlp"]["model"].predict_proba(X_test_scaled)[:, 1]
    mlp_auc_test = get_auc_score(y_test, mlp_y_probas)
    mlp_acc_test = accuracy_score(y_test, mlp_y_probas >= 0.5)

    results["data"].update({"X_test": X_test, "y_test": y_test})
    results["logreg"].update({"acc_test": logreg_acc_test, "auc_test": logreg_auc_test})
    results["mlp"].update({"acc_test": mlp_acc_test, "auc_test": mlp_auc_test})

    results.update(
        {
            "best_solution": best_solution,
            "args": args,
            "scaler": results["data"]["scaler"],
            "time": time.time() - start_time,
        }
    )
    return results


def get_auc_score(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)


def print_test_results(results):
    if "data" in results:
        y_test = results["data"]["y_test"]
        print(f"Test set size: {len(y_test)}")
    logreg_acc_test = results["logreg"]["acc_test"]
    logreg_auc_test = results["logreg"]["auc_test"]
    print(f"Logreg acc test: {logreg_acc_test:.1%}, auc test: {logreg_auc_test:.3f}")
    mlp_acc_test = results["mlp"]["acc_test"]
    mlp_auc_test = results["mlp"]["auc_test"]
    print(f"MLP acc test: {mlp_acc_test:.1%}, auc test: {mlp_auc_test:.3f}")
    print("Elapsed time", results["time"])
    return results


def aggregate_results_targeted(results, models):
    for model in models:
        acc_train = np.mean(results[model]["accs_train"])
        acc_eval = np.mean(results[model]["accs_eval"])
        acc_test = np.mean(results[model]["accs_test"])
        auc_test = np.mean(results[model]["aucs_test"])
        results[model].update(
            {
                "acc_train": acc_train,
                "acc_eval": acc_eval,
                "acc_test": acc_test,
                "auc_test": auc_test,
            }
        )
        print(
            f"{model}: acc train (mean)={acc_train:.2%}; acc eval (mean)={acc_eval:.2%}; acc test (mean)={acc_test:.2%} auc test (mean)={auc_test:.3f}"
        )
    results["time"] = np.mean(results["times"])
    results["mean_num_queries_shadow_qbs"] = np.mean(results["num_queries_shadow_qbs"])
    print(
        "Elapsed time (mean)",
        results["time"],
        "Number of queries/shadow qbs",
        results["mean_num_queries_shadow_qbs"],
    )
    return results


def load_or_generate_results(
    args,
    save_dir,
    aux_split,
    test_split,
    tar_user,
    tar_qbs_seeds,
    tar_sensitive_attributes,
    seed,
):
    prefix = get_prefix(args)
    save_path_r = f"{save_dir}/tar-{args.target_idx}_{prefix}_seed-{seed}"
    results_path = f"{save_path_r}_results.pickle"
    if os.path.exists(results_path):
        print("Loading the results from", results_path)
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    else:
        # Note that the best hyperparameters selected based on the
        # validation data.
        print("Generating the results and saving them to", results_path)
        results = run_query_search(
            args,
            save_path_r,
            aux_split,
            test_split,
            tar_user,
            tar_qbs_seeds,
            tar_sensitive_attributes,
            seed,
        )
        with open(results_path, "wb") as f:
            pickle.dump(results, f)

    return results


def main_experiment(save_dir, args):
    # Load the data.
    data = DatasetLoader(args.dataset_path, args.dataset_name)

    np.random.seed(args.seed)
    # Sample one seed for each repetition.
    seeds = np.random.choice(10**8, 100, replace=False)
    # print(seeds)

    seed = seeds[args.repetition]
    np.random.seed(seed)
    print("Repetition #", args.repetition, "Seed:", seed)

    attributes = data.sample_attributes(num_attributes=args.num_attributes)

    # The data will be split into two parts: auxiliary and target splits,
    # where the size of the target split is one third of the dataset size.
    test_split_size = len(data.all_records) // 3
    print("test_split_size", test_split_size)

    data.split_dataset(test_size=test_split_size, aux_size=None)
    aux_split, aux_idxs = data.get_auxiliary_split()
    print(f"Size of the auxiliary split: {len(aux_split)}")
    test_split, test_idxs = data.get_test_split()
    print(f"Size of the test split: {len(test_split)}")

    # The target users are selected among the unique users in the test split.
    unique_users = get_indexes_unique(test_split)
    print("Number of unique records in the test split", len(unique_users))
    tar_users = unique_users[: args.num_target_users]
    # print('tar_users', tar_users, test_split[tar_users])

    # Seeds used to initialize the random number generator of a QBS instance.
    tar_qbs_seeds = np.random.choice(10**8, size=args.num_test_samples, replace=False)
    # print('tar_qbs_seeds', tar_qbs_seeds)

    # Target sensitive attribute values for the `exact` scenario, where the
    # target dataset is constant, except for the sensitive attribute of the
    # target record.
    tar_sensitive_attributes = np.random.permutation(
        np.tile([0, 1], args.num_test_samples // 2)
    )
    if args.num_test_samples % 2 == 1:
        tar_sensitive_attributes = np.append(
            tar_sensitive_attributes, np.random.randint(2)
        )
    assert len(tar_sensitive_attributes) == args.num_test_samples
    # print('tar_sensitive_attributes', tar_sensitive_attributes)

    all_results = dict()
    # Targeted attack: run one query search per target user.
    print(f"Running a targeted attack against {len(tar_users)} users.")
    models = ["logreg", "mlp"]
    eval_types = ["train", "eval", "test"]
    results = {
        model: {f"accs_{eval_type}": [] for eval_type in eval_types} for model in models
    }
    for model in models:
        results[model]["aucs_test"] = []
    results["times"] = []
    results["num_queries_shadow_qbs"] = []
    ## To enforce more variability in experiments ran for different users,
    ## uncomment the statement below and replace `seed` in args_list with
    ## `seed_targets[ti]`.
    ## The current seed is the same across target users. Since in the AUXILIARY
    ## scenario, the target user may not belong to the auxiliary split, all the
    ## users for which this holds will have the same shadow datasets, same
    ## query search randomness. However, the best solution in the first
    ## generation and the solutions afterwards will vary.
    ## This does not significantly impact the final results (numerically); we
    ## however believe that future work extending this code should enforce the
    ## use of different randomness for different users in the same run.
    ## Code to uncomment below:
    # seed_targets = np.random.choice(10**8, size=len(tar_users)).
    for ti_start in range(0, len(tar_users), args.num_procs):
        ti_end = min(ti_start + args.num_procs, len(tar_users))
        # print(ti_start, ti_end)
        args_list = []
        for ti in range(ti_start, ti_end, 1):
            args_ti = copy.deepcopy(args)
            args_ti.target_idx = ti
            # Disable parallelism during the search.
            args_ti.num_procs = 1
            args_list.append(
                (
                    args_ti,
                    save_dir,
                    aux_split,
                    test_split,
                    tar_users[ti],
                    tar_qbs_seeds,
                    tar_sensitive_attributes,
                    seed,
                )
            )
        with multiprocessing.Pool(args.num_procs) as pool:
            results_batch = pool.starmap(load_or_generate_results, args_list)
        for ti, results_tar_user in enumerate(results_batch):
            print(f"Results for the target user {ti_start+ti}")
            print_test_results(results_tar_user)
            for model in models:
                for eval_type in eval_types:
                    results[model][f"accs_{eval_type}"].append(
                        results_tar_user[model][f"acc_{eval_type}"]
                    )
                results[model]["aucs_test"].append(results_tar_user[model]["auc_test"])
            results["times"].append(results_tar_user["time"])
            results["num_queries_shadow_qbs"].append(
                results_tar_user["num_queries_shadow_qbs"]
            )

    results = aggregate_results_targeted(results, models)

    prefix = get_prefix(args)

    all_results.update(
        {
            "results": results,
            "attributes": attributes,
            "test_idxs": test_idxs,
            "auxiliary_idxs": aux_idxs,
            "target_users": tar_users,
            "args": args,
            "seed": seed,
            "target_qbs_seeds": tar_qbs_seeds,
        }
    )

    if args.dataset_sampler == "exact":
        all_results["target_sensitive_attributes"] = tar_sensitive_attributes

    with open(
        f"{save_dir}/sol-1-na-{args.num_attributes}_{prefix}_seed-{seed}.pickle", "wb"
    ) as f:
        pickle.dump(all_results, f)
