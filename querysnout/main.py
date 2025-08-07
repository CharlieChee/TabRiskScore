import argparse
import os
import time

from src.experiments import main_experiment


def str2bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser(description="Automated query attack.")
    # Experiment to run.
    parser.add_argument("--experiment", type=str, default="main")
    parser.add_argument("--repetition", type=int, default=0)
    # General.
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--verbose", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_procs", type=int, default=4)
    # Path for the dataset.
    parser.add_argument("--dataset_path", type=str, default="datasets/adults")
    parser.add_argument("--dataset_name", type=str, default="adults")
    parser.add_argument("--target_dataset_size", type=int, default=8000)
    # Method to sample datasets (can be "without_replacement" or "exact").
    parser.add_argument("--dataset_sampler", type=str, default="without_replacement")
    parser.add_argument("--num_target_users", type=int, default=100)
    # For targeted attacks, how many qbs instances to use for testing.
    parser.add_argument("--num_test_samples", type=int, default=500)
    # QBS environment.
    parser.add_argument("--num_attributes", type=int, default=5)
    parser.add_argument("--num_datasets", type=int, default=3000)
    # Fraction of the datasets to use for evaluation.
    parser.add_argument("--eval_fraction", type=float, default=0.33333)
    # QBS parameters.
    # The QBS type can be "simple", "table-builder" or "dp-laplace".
    # Cf. src/optimized_qbs/qbs.c for instructions relating "diffix".
    parser.add_argument("--qbs_type", type=str, default="simple")
    parser.add_argument("--qbs_threshold", type=int, default=4)
    parser.add_argument("--qbs_noise_scale", type=float, default=3.0)
    parser.add_argument("--qbs_epsilon", type=float, default=5.0)
    # Query search: "evolutionary" or "random".
    parser.add_argument("--search_type", type=str, default="evolutionary")
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--num_queries", type=int, default=100)
    # Mutation parameters.
    parser.add_argument("--p_copy", type=float, default=0.025)
    parser.add_argument("--p_modify", type=float, default=0.025)
    parser.add_argument("--p_swap", type=float, default=1 / 6)
    # Note: The term "change" in the paper and "switch" here refer to the same
    # thing.
    parser.add_argument("--p_switch", type=float, default=1 / 6)

    parser.add_argument("--frac_elitism", type=float, default=0.1)
    parser.add_argument("--model_type", type=str, default="logreg")
    parser.add_argument("--num_generations", type=int, default=200)
    return parser


def check_args(args):
    assert len(args.save_dir) > 0, "ERROR: Empty save directory."
    assert (
        args.eval_fraction > 0 and args.eval_fraction < 1
    ), "ERROR: Invalid fraction of evaluation samples."
    assert args.search_type in [
        "random",
        "evolutionary",
    ], f"ERROR: Invalid search type {args.search_type}."
    assert (
        args.frac_elitism > 0 and args.frac_elitism < 1
    ), "ERROR: Invalid fraction of elites."
    assert args.model_type in ["logreg", "mlp"], "ERROR: Invalid model type."
    assert args.dataset_sampler in [
        "exact",
        "without_replacement",
    ], f"ERROR: Invalid dataset sampler {args.dataset_sampler}."
    assert (
        args.num_target_users > 0
    ), f"ERROR: Invalid number of target users {args.num_target_users}, should be larger than 0."
    assert args.qbs_type in [
        "diffix",
        "simple",
        "table-builder",
        "dp-laplace",
    ], "ERROR: QBS should be either `diffix`, `simple` or `table-builder`."
    assert args.qbs_threshold >= 0, "ERROR qbs_threshold must be non-negative."
    assert args.qbs_noise_scale >= 0, "ERROR: qbs_noise_scale must be non-negative."
    assert args.qbs_epsilon > 0, "ERROR: qbs_epsilon must be strictly positive."


if __name__ == "__main__":
    start_time = time.time()
    args = get_parser().parse_args()
    check_args(args)

    print(args)

    if args.dataset_sampler == "exact":
        ds = "ds-exact"
    elif args.dataset_sampler == "without_replacement":
        ds = "ds-without-replacement"
    else:
        raise ValueError(f"Invalid dataset sampler {args.dataset_sampler}")
    ds += f"-targeted-{args.num_target_users}"

    if args.qbs_type == "simple":
        qbs = f"qbs-simple-[thr={args.qbs_threshold}_ns={args.qbs_noise_scale}]"
    elif args.qbs_type == "diffix":
        qbs = "qbs-diffix"
    elif args.qbs_type == "table-builder":
        qbs = "qbs-table-builder"
    elif args.qbs_type == "dp-laplace":
        qbs = f"qbs-dp-laplace-[eps={args.qbs_epsilon}]"
    else:
        raise ValueError(f"Invalid qbs type {args.qbs_type}")

    save_dir = os.path.join(args.save_dir, args.dataset_name, qbs, ds)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Save directory: ", save_dir)

    main_experiment(save_dir, args)
    end_time = time.time()
    print(f"Elapsed time: {end_time-start_time} secs.")
