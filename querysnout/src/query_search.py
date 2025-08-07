import multiprocessing
import time
import warnings
from collections import namedtuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .helpers.nice import display_solution
from .helpers.utils import add_occurrences_to_list, list_to_pdf


class QuerySearch:
    """
    Search for a set of queries such that their answers (as provided by
    a query-based system) for a given user, when combined, allow the retrieval
    of the user's sensitive attribute. The sensitive attribute is typically
    protected by the query-based system via noise addition.

    The query search is based on genetic algorithms. Each solution consists of
    a set of queries. The fitness of a solution is defined as the accuracy of
    an ML model to predict a user's sensitive attribute. The ML model is fitted
    on query answers for many users, and a wide range of datasets.

    The search space is composed of all sets of queries of size `num_queries`.
    A query is of type "SELECT COUNT WHERE cond1 AND ... condi".
    There can be at most one condition per attribute. Each condition is of the
    form "==", "!=", or "no condition".

    A query-based environment handles the querying process efficiently.

    This is a partially abstract class, that implements the fitness (querying,
    maintaining fitness arrays) and population tracking, as well as elitism
    (which can be disabled), but *not* mutations.
    """

    def __init__(
        self,
        qbs_environment,
        num_attributes,
        population_size,
        num_queries,
        frac_elitism,
        model_type="logreg",
        num_procs=4,
    ):
        """
        Initializes the algorithm parameters and the population.

        qbs_environment: QBSEnvironment object.
            The QBS + Dataset(s) on which the attack is performed.
            This is treated as a queryable black-box.
        population_size: int.
            Size of the genetic algorithm's population (number of solutions).
        num_queries: int.
            Number of queries in a solution.
        frac_elitism: float in [0,1].
            Elitism: fraction of the population's best individuals to transfer
            untouched to the next generation.
        model_type: string ("logreg", or "mlp").
            Which Machine Learning model to use as an attack when evaluating the
            fitness.
        num_procs: int.
            Number of processes used to parallelize the fitness evaluation.
            Set it to 1 to compute the fitness sequentially.
        """
        # The qbs environment (to be queried in a black-box fashion).
        self.qbs_environment = qbs_environment
        self.num_attributes = num_attributes
        self.population_size = population_size
        # Number of queries in the solution.
        self.num_queries = num_queries
        # Degree of elitism, given as the fraction of best solutions passed as
        # is to the next generation.
        self.frac_elitism = frac_elitism
        self.num_elites = int(self.frac_elitism * self.population_size)
        print("Number of elites", self.num_elites)
        # Type of the machine learning model (fitness).
        self.model_type = model_type
        self.num_procs = num_procs

        # Population of solutions.
        self.population = []
        # Fitnesses of solutions (stored in same order as in the population).
        self.fitnesses_train = []
        self.fitnesses_eval = []
        self.fitnesses_min = []

        # For time reasons we will use a small number of iterations for the
        # logistic regression, which is likely to trigger warnings.
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def random_query(self):
        """Returns a random query (tuple of -1,0,+1 of length num_attributes)."""
        return tuple(np.random.randint(-1, 2, self.num_attributes))

    def init_solution(self):
        """
        Initializes a solution uniformly at random in the search space.
        """
        return [self.random_query() for _ in range(self.num_queries)]

    def sort_population(self):
        """Sort (in-place) the current population by decreasing fitness."""
        sorted_idxs = np.argsort(self.fitnesses_min)[::-1]
        self.fitnesses_min = [self.fitnesses_min[i] for i in sorted_idxs]
        self.population = [self.population[i] for i in sorted_idxs]
        self.fitnesses_eval = [self.fitnesses_eval[i] for i in sorted_idxs]
        self.fitnesses_train = [self.fitnesses_train[i] for i in sorted_idxs]

    def init_population(self, verbose=False):
        """Initializes the population randomly."""
        self.population = [self.init_solution() for _ in range(self.population_size)]
        print(f"Initialized a population of size {self.population_size}")

        self.eval_population(verbose=verbose)

        print(f"Average fitness: {np.mean(self.fitnesses_eval):.1%}.")

    def format_solution(self, solution):
        if self.qbs_environment.qbs_type == "dp-laplace":
            # Group the repeated queries and accumulate their budget. For
            # instance if a query is repeated twice, we represent it as a
            # single query with the double budget.
            return list_to_pdf(solution)
        elif self.qbs_environment.deterministic:
            return solution
        else:
            return add_occurrences_to_list(solution)

    def eval_population(self, starting_index=0, verbose=False):
        """
        Evaluates each solution in the population, starting from the
        starting_index position, then orders all solutions decreasingly
        by fitness.
        """
        # TODO: parallelize this?
        # /!\ We can't just simply run a pool.map on this, since it messes up
        #  with the memoization. Restructuring the code would be needed.
        fitnesses_train = self.fitnesses_train[:starting_index]
        fitnesses_eval = self.fitnesses_eval[:starting_index]

        queries = set()
        population_formatted = [
            self.format_solution(solution)
            for solution in self.population[starting_index:]
        ]
        for solution in population_formatted:
            queries.update(solution)

        self.qbs_environment.update_cache(queries)
        y_train, y_eval = self.qbs_environment.get_labels()

        # Run the fitness evaluation in parallel.
        args_to_process = []
        for i, solution in enumerate(population_formatted):
            # start_time = time.time()
            X_train, X_eval = self.qbs_environment.get_answers(solution)
            # print('Time to retrieve and concat the answers', time.time() - start_time)
            args_to_process.append(
                (i, X_train, y_train, X_eval, y_eval, verbose, self.model_type)
            )

        if self.num_procs > 1:
            with multiprocessing.Pool(self.num_procs) as pool:
                results = pool.map(
                    QuerySearch._compute_fitness_parallel, args_to_process
                )
        else:
            results = [
                QuerySearch._compute_fitness_parallel(args_to_process[i])
                for i in range(len(args_to_process))
            ]

        for fitness_train, fitness_eval in results:
            fitnesses_train.append(fitness_train)
            fitnesses_eval.append(fitness_eval)

        self.fitnesses_train = fitnesses_train
        self.fitnesses_eval = fitnesses_eval
        # EXPERIMENT (prevent overfitting either way).
        self.fitnesses_min = np.minimum(self.fitnesses_train, self.fitnesses_eval)

        self.sort_population()

    def fitness(self, solution, verbose=False, model_type=None):
        """
        Evaluate the fitness of a given solution.

        The fitness of a solution is the accuracy of an attacker using
        a ML model (defined by `model_type`) on the queries (given by
        the `solution`), trained on a set of unique users (and datasets),
        and evaluated on other users (potentially from other datasets).

        By default (model_type=None), this uses self.model_type. You may
        specify model_type to either `mlp` or `logreg` to force evaluation
        with a specific model.
        """

        if model_type is None:
            model_type = self.model_type
        # STEP 1: QUERIES.
        # Call the QBS to get the query answers.
        solution = self.format_solution(solution)
        X_train, X_eval = self.qbs_environment.get_answers(solution)
        y_train, y_eval = self.qbs_environment.get_labels()
        # print(X_train, y_train)

        return QuerySearch._compute_fitness(
            X_train, y_train, X_eval, y_eval, verbose, model_type
        )

    def _compute_fitness_parallel(args):
        # start_time = time.time()
        pid, X_train, y_train, X_eval, y_eval, verbose, model_type = args
        if verbose is True:
            print(f"Starting process {pid}")
        return QuerySearch._compute_fitness(
            X_train, y_train, X_eval, y_eval, verbose, model_type
        )

    def _compute_fitness(X_train, y_train, X_eval, y_eval, verbose, model_type):
        # STEP 2: MACHINE LEARNING.
        # Scale the query answers to facilitate learning
        scaler = StandardScaler().fit(X_train)

        # TODO: could this model be passed as an argument of the class?
        if model_type == "logreg":
            model = LogisticRegression(max_iter=200)
        elif model_type == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(50, 20), early_stopping=True)
        else:
            raise ValueError(
                "ERROR: Model should be either Logistic ", "Regression or MLP"
            )

        X_train_scaled = scaler.transform(X_train)
        model = model.fit(X_train_scaled, y_train)  # Bottleneck.
        # print(X_train_scaled, y_train)
        # if type(model) == MLPClassifier:
        #    print('MLP', model.coefs_)
        # elif type(model) == LogisticRegression:
        #    print('LogisticRegression', model.coef_)
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Elapsed time for {model_type}: {elapsed_time:.5f} secs")
            # print('Means', scaler.mean_, 'Stds', scaler.scale_)

        # Evaluate the success
        # TODO: pass a custom score to the class?
        fitness_train = model.score(X_train_scaled, y_train)
        fitness_eval = model.score(scaler.transform(X_eval), y_eval)

        return fitness_train, fitness_eval

    def selection(self, num_parents):
        """Randomly select parents from the population for the next generation."""
        # The sampling probability is proportional to its fitness.
        parent_idxs = np.random.choice(
            self.population_size,
            replace=True,
            size=num_parents,
            p=self.fitnesses_min / np.sum(self.fitnesses_min),
        )
        # TODO: *maybe* give the fitness->proba function as argument?
        return [self.population[i] for i in parent_idxs]

    def mutation(self, parent):
        """
        Mutation of a parent into an offspring.

        First, one of the following is chosen at random: (1) adding a query,
        (2) deleting a query, or (3) replacing one or more queries.

        If (1) or (2) are picked, a new query is sampled uniformly at random.
        """
        # By default, does nothing.
        return parent.copy()

    def generation(self, verbose):
        """
        Create a new generation via mutation of the current population.
        """
        # Pass a fraction of the population (the elites) onto the next
        # generation, unchanged.
        generation = self.population[: self.num_elites]
        # For each remaining spot, select a random parent in the population,
        # then mutate that parent to get an offspring.
        parents = self.selection(self.population_size - self.num_elites)
        for parent in parents:
            generation.append(self.mutation(parent))

        self.population = generation
        # self.display_population(3)

        self.eval_population(starting_index=self.num_elites, verbose=verbose)

        return np.mean(self.fitnesses_train), np.mean(self.fitnesses_eval)

    def display_population(self, num_to_display):
        for i, solution in enumerate(self.population[:num_to_display]):
            display_solution(solution)
        # for i, solution in enumerate(self.population):
        #    print(i, len(solution), self.fitnesses_eval[i])


class RandomQuerySearch(QuerySearch):
    """QuerySearch instance that fully mutates the population at each
    generation, except for the top solution (elite)."""

    def __init__(
        self,
        qbs_environment,
        num_attributes,
        population_size,
        num_queries,
        model_type="logreg",
        num_procs=4,
    ):
        """Initialize the object [see QuerySearch for inherited parameters]."""
        # Manually set the frac_elitism to keep the top model (elite) each generation.
        frac_elitism = 1 / population_size  # Keep only one elite.
        QuerySearch.__init__(
            self,
            qbs_environment,
            num_attributes,
            population_size,
            num_queries,
            frac_elitism,
            model_type,
            num_procs,
        )

    def mutation(self, parent):
        """Fully mutate the parent (random search)"""
        return [self.random_query() for _ in range(self.num_queries)]


MutationProbabilities = namedtuple(
    "MutationProbabilities", ["p_copy", "p_modify", "p_switch", "p_swap"]
)


class EvolutionaryQuerySearch(QuerySearch):
    """QuerySearch instance that implements simple mutations."""

    def __init__(
        self,
        qbs_environment,
        num_attributes,
        population_size,
        num_queries,
        frac_elitism,
        mutation_probs,
        model_type="logreg",
        num_procs=4,
    ):
        """
        Create the object [see QuerySearch for inherited parameters].
        Additional parameters control the mutation behaviour.

        mutation_probs: MutationProbabilities.
            Specifies the probabilities used to mutate a solution, such as
            copying or changing queries.
        """
        QuerySearch.__init__(
            self,
            qbs_environment,
            num_attributes,
            population_size,
            num_queries,
            frac_elitism,
            model_type,
            num_procs,
        )
        # Mutation probabilties.
        self.p_copy = mutation_probs.p_copy
        self.p_modify = mutation_probs.p_modify
        self.p_switch = mutation_probs.p_switch
        self.p_swap = mutation_probs.p_swap

    def modify_query(self, query):
        swap_idxs = []
        switch_idxs = []

        shuffled_idxs = np.random.permutation(len(query))
        curr_i = 0
        while curr_i < len(shuffled_idxs):
            modif = np.random.choice(
                3, p=(self.p_switch, self.p_swap, 1 - self.p_switch - self.p_swap)
            )
            if modif == 0:
                # Switch.
                switch_idxs.append(shuffled_idxs[curr_i])
                curr_i += 1
            elif modif == 1:
                # Swap with another random column.
                # If the last index, skip it.
                if curr_i == len(shuffled_idxs) - 1:
                    curr_i += 1
                else:
                    swap_idxs.append((shuffled_idxs[curr_i], shuffled_idxs[curr_i + 1]))
                    curr_i += 2
            else:
                # Leave the condition unchanged.
                curr_i += 1

        # print('Query', query)
        # print('Swap indexes', swap_idxs)
        # print('Switch indexes', switch_idxs)

        mutated_query = list(query)
        for i, j in swap_idxs:
            mutated_query[i] = query[j]
            mutated_query[j] = query[i]
        for i in switch_idxs:
            # Switching the current condition to a different value (among -1,
            # 0, or 1).
            mutated_query[i] = (query[i] + np.random.randint(1, 3)) % 3
            # Converting 2 to -1.
            if mutated_query[i] == 2:
                mutated_query[i] = -1

        # print('Query', query, 'Mutated query', mutated_query)
        return tuple(mutated_query)

    def mutation(self, parent):
        """
        Mutation of a parent into an offspring.

        Each query is copied or modified according to the specified mutation
        probabiities. The copied/modified query is changed randomly and
        incrementally by either switching a condition or swapping the existing
        conditions between two columns.
        """
        offspring = []

        mutations = np.random.choice(
            3,
            size=len(parent),
            p=(self.p_copy, self.p_modify, 1 - self.p_copy - self.p_modify),
        )

        # print('Mutations', mutations)

        for i, mutation in enumerate(mutations):
            if mutation == 0:
                # Copy.
                offspring.append(parent[i])
                # If the system is deterministic we always modify the copy.
                # Otherwise, we modify the query with probability 0.5 and
                # it modify with probability 0.5.
                if self.qbs_environment.deterministic or np.random.randint(0, 2) == 0:
                    offspring.append(self.modify_query(parent[i]))
                else:
                    offspring.append(parent[i])
            elif mutation == 1:
                # Modify.
                offspring.append(self.modify_query(parent[i]))
            else:
                # Leave unchanged.
                offspring.append(parent[i])

        # Randomly permute the solution, which could exceed the maximum
        # query size.
        np.random.shuffle(offspring)

        # Keep up to num_queries queries.
        offspring = offspring[: self.num_queries]

        return offspring
