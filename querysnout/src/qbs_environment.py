import multiprocessing
import time
from collections import Counter

import numpy as np

from .helpers.utils import init_qbs


class QBSEnvironment:
    """
    Environment consisting of several Query-Based Systems (QBS), each answering
    queries on real datasets.

    Given a solution consisting of a set of queries, evaluates the solution's
    performance in predicting the correct value of the sensitive attribute.
    The performance is estimated by first fitting a machine learning
    classifier on a split of training users, then computing the accuracy on a
    split of unseen users.

    The environment contains options for using the same datasets, or different
    datasets for fitting and evaluation.
    """

    def __init__(
        self,
        dataset_sampler,
        num_datasets,
        eval_fraction,
        qbs_type,
        qbs_threshold,
        qbs_noise_scale,
        qbs_epsilon,
        verbose=True,
        num_procs=1,
        qbs_seeds_to_exclude=[],
    ):
        """
        Initializes the datasets and the training/evaluation splits.

        dataset_sampler: DatasetSampler. Instance of DatasetSampler used to
            sample datasets.
        num_datasets: int. Total number of datasets used (for train).
        eval_fraction: int.
            Fraction of users/datasets used in the evaluation.
        qbs_type: str ("diffix" or "simple")
            Type of QBS to use, between Diffix and Simple.
        qbs_threshold: int
            For Simple QBS, the threshold for bucket suppression.
        qbs_noise_scale: float
            For Simple QBS, the scale of the Gaussian noise added.
        qbs_epsilon: float
            For DPLaplace, the privacy budget.
        verbose: boolean.
            Whether to display timers.
        qbs_seeds_to_exclude: int.
            The seeds for the randomness of QBSes should be distinct from these
            values (which are used by the test QBSes).
        num_procs: int.
            The number of processes to use when querying. Set it to 1 to query
            the QSBes sequentially.
        """
        self.dataset_sampler = dataset_sampler
        self.num_datasets = num_datasets
        self.eval_fraction = eval_fraction
        self.qbs_type = qbs_type
        self.qbs_threshold = qbs_threshold
        self.qbs_noise_scale = qbs_noise_scale
        self.qbs_epsilon = qbs_epsilon
        self.verbose = verbose
        self.num_procs = num_procs
        # Set of qbs seeds to exclude. This way, we ensure that the shadow
        # QBSes are initialized with different seeds than the test QBSes.
        self.qbs_seeds_to_exclude = set(qbs_seeds_to_exclude)
        # Sample datasets by the given technique.
        self.datasets = []
        self.idxs_users = []

        self._sample_datasets()

        if self.verbose:
            self._print_dataset_summary()

        self._split_train_eval()

        self._init_qbs()

        self._init_cache()

    def _init_cache(self):
        # Initialize the cache of query answers. For each dataset, the cache
        # is a dictionary, mapping a query to the answers to that query
        # for users in the train and/or eval split. The answers are represented
        # as a Nx1 array, where N is the number of users in the split. The
        # i-th row contains the answer to the query for the i-th user in
        # self.split_idxs.
        self.query_answers = [
            {"train": {}, "eval": {}} for _ in range(len(self.datasets))
        ]
        self._cached_queries = set()

    def _print_dataset_summary(self):
        counter = Counter([len(dataset) for dataset in self.datasets])
        print(f"Summary of the dataset sizes:", counter)
        print(
            f"Histogram of target idxs",
            np.histogram([idxs[0] for idxs in self.idxs_users]),
        )

    def _init_qbs(self):
        # Initialize the query-based systems.
        self.qbs = []
        seeds = set()
        # Sampling seeds that are different from the seeds of the test QBS
        # instances.
        num_trials = 0
        while len(seeds) < self.num_datasets:
            num_trials += 1
            seed = np.random.randint(10**8)
            if seed in self.qbs_seeds_to_exclude or seed in seeds:
                continue
            else:
                seeds.add(seed)
        seeds = list(seeds)
        assert len(seeds) == self.num_datasets
        # print(seeds)
        print(
            f"Done sampling {self.num_datasets} different seeds (for the shadow QBSes) in {num_trials} trials."
        )

        for i, dataset in enumerate(self.datasets):
            qbs = init_qbs(
                dataset,
                self.qbs_type,
                self.qbs_threshold,
                self.qbs_noise_scale,
                self.qbs_epsilon,
                seed=seeds[i],
            )
            self.qbs.append(qbs)
        # Will cache the results of the queries when the qbs is deterministic.
        if self.qbs_type == "diffix":
            self.deterministic = True
        elif self.qbs_type == "simple" and self.qbs_noise_scale == 0:
            self.deterministic = True
        elif self.qbs_type == "table-builder":
            self.deterministic = True
        elif self.qbs_type == "dp-laplace":
            self.deterministic = False
        else:
            self.deterministic = False

    def _split_train_eval(self):
        # The train/eval split is between *datasets*.
        # The indices of training and evaluation datasets.
        self.train_didxs = np.arange(int(len(self.datasets) * (1 - self.eval_fraction)))
        self.eval_didxs = np.arange(
            int(len(self.datasets) * (1 - self.eval_fraction)), len(self.datasets)
        )
        # The training and evaluation users come from different
        # datasets.
        self.train_idxs = [self.idxs_users[di] for di in self.train_didxs]
        self.eval_idxs = [self.idxs_users[di] for di in self.eval_didxs]
        # print('train_didxs', self.train_didxs)
        # print('train_idxs', self.train_idxs)
        # print('eval_didxs', self.eval_didxs)
        # print('eval_idxs', self.eval_idxs)

        # The sensitive attribute is the last column of each dataset.
        self.y_train = np.concatenate(
            [
                self.datasets[di][self.train_idxs[i], -1]
                for i, di in enumerate(self.train_didxs)
            ]
        )
        self.y_eval = np.concatenate(
            [
                self.datasets[di][self.eval_idxs[i], -1]
                for i, di in enumerate(self.eval_didxs)
            ]
        )

        print(
            f"Size of training set: {len(self.y_train)}, eval set: {len(self.y_eval)}"
        )
        print(
            f"Fraction of positive labels - train: {np.sum(self.y_train)/len(self.y_train):.2f} / eval: {np.sum(self.y_eval)/len(self.y_eval):.2f}"
        )

    def _sample_datasets(self):
        start_time = time.time()

        for di in range(self.num_datasets):
            if self.dataset_sampler.name == "auxiliary":
                if di < int(self.num_datasets * (1 - self.eval_fraction)):
                    eval_type = "train"
                else:
                    eval_type = "eval"
                dataset, idxs_users = self.dataset_sampler.sample_dataset(eval_type)
            else:
                dataset, idxs_users = self.dataset_sampler.sample_dataset()
            self.idxs_users.append(list(idxs_users))
            # if di == 0:
            #    print('Dataset train', dataset)
            # elif di == self.num_datasets - 1:
            #    print('Dataset eval', dataset)
            self.datasets.append(dataset.astype(int))

        elapsed_time = time.time() - start_time
        print(
            f"Done initializing the {self.num_datasets} datasets by the ",
            f"{self.dataset_sampler} technique. Elapsed time (secs): ",
            f"{elapsed_time:.0f}",
        )

    def _query_runner(args):
        """Performs a query on an individual process."""
        qbs, indices, queries, budgets, di = args
        # print(qbs, indices, queries, budgets, di)
        if budgets is not None:
            answers = qbs.structured_query(indices, queries, budgets)
        else:
            answers = qbs.structured_query(indices, queries)
        return np.array(answers).reshape(-1, len(queries))

    def update_cache(self, queries):
        """
        Initializes the query cache with the answer to each query. The current
        version exploits the fact that Diffix is a deterministic query-based
        system. TODO: Update the procedure in order to account for
        non-deterministic noise addition.

        The procedure is parallelized over the datasets.
        """

        # Select the queries that have not been answered before.
        new_queries = [q for q in queries if q not in self._cached_queries]
        self._cached_queries.update(new_queries)

        if len(new_queries) == 0:
            return 0
        print(f"\nQuerying in parallel for {len(new_queries)} queries...")
        # Setup the training/testing queries to call (with multiprocessing).
        queries_to_process = []
        if self.qbs_type == "dp-laplace":
            new_queries_clean, budgets = zip(*new_queries)
            new_queries_clean, budgets = list(new_queries_clean), list(budgets)
        elif not self.deterministic:
            new_queries_clean, budgets = [q[0] for q in new_queries], None
        else:
            new_queries_clean, budgets = new_queries, None
        for i, di in enumerate(self.train_didxs):
            queries_to_process.append(
                (self.qbs[di], self.train_idxs[i], new_queries_clean, budgets, di)
            )
        for i, di in enumerate(self.eval_didxs):
            queries_to_process.append(
                (self.qbs[di], self.eval_idxs[i], new_queries_clean, budgets, di)
            )

        if self.num_procs > 1:
            # Run these queries in multi-processing.
            with multiprocessing.Pool(self.num_procs) as pool:
                all_answers = pool.map(QBSEnvironment._query_runner, queries_to_process)
        else:
            all_answers = [
                QBSEnvironment._query_runner(queries_to_process[i])
                for i in range(len(queries_to_process))
            ]

        # Split these answers in training and evaluation.
        # Each answer corresponds to *all* the queries run on a single dataset
        # (QBS).
        split = len(self.train_didxs)
        train_answers, eval_answers = all_answers[:split], all_answers[split:]
        # Memoize these answers for the rest.
        for i, (di, answers) in enumerate(zip(self.train_didxs, train_answers)):
            self.query_answers[di]["train"].update(
                {q: answers[:, j].reshape(-1, 1) for j, q in enumerate(new_queries)}
            )

        for i, (di, answers) in enumerate(zip(self.eval_didxs, eval_answers)):
            self.query_answers[di]["eval"].update(
                {q: answers[:, j].reshape(-1, 1) for j, q in enumerate(new_queries)}
            )

        return len(new_queries)

    def get_answers(self, queries):
        """
        Given a set of cached queries, retrieves the answers for each dataset,
        and returns them separated into train and evaluation.
        """
        # Collect all the query answers.
        X_train = np.concatenate(
            [
                np.concatenate(
                    [self.query_answers[di]["train"][q] for q in queries], axis=1
                )
                for di in self.train_didxs
            ],
            axis=0,
        )
        X_eval = np.concatenate(
            [
                np.concatenate(
                    [self.query_answers[di]["eval"][q] for q in queries], axis=1
                )
                for di in self.eval_didxs
            ],
            axis=0,
        )

        return X_train, X_eval

    def get_labels(self):
        return self.y_train, self.y_eval
