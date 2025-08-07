import pickle
import time

import matplotlib.pyplot as plt
import numpy as np


class GenerationLogger:
    """Class that tracks and displays informations from generations."""

    def __init__(self, query_search, save_path):
        """
        Register the query search to monitor, and initialise.

        query_search: QuerySearch object.
            The object to monitor.
        save_path: string.
            Path (directory/prefix) to which to save the log of results and
            the plots.
        """
        self.query_search = query_search
        self._start_time = time.time()
        self._last_tic = self._start_time
        self.times = []
        self.save_path = save_path
        # Log, for future plots.
        self.mean_fitnesses = {"x": [], "train": [], "eval": []}
        self.max_fitnesses = {"x": [], "train": [], "eval": []}
        self.complete = False

    def log(self, generation):
        """Log the current progress."""
        self.mean_fitnesses["x"].append(generation)
        self.max_fitnesses["x"].append(generation)
        for fitness, target in zip(
            [self.query_search.fitnesses_train, self.query_search.fitnesses_eval],
            ["train", "eval"],
        ):
            fmean = np.mean(fitness)
            fmax = np.max(fitness)
            self.mean_fitnesses[target].append(fmean)
            self.max_fitnesses[target].append(fmax)
        # Nice display, ugly code.
        BOLD = "\033[1m"
        ENDC = "\033[0m"
        text = (
            f"\r[{generation}]: {BOLD}max fitness = {100*fmax:.1f}{ENDC} ",
            f"(train: {100*self.max_fitnesses['train'][-1]:.1f}), ",
            f"{BOLD}mean fitness = {100*fmean:.1f}{ENDC} ",
            f"(train: {100*self.mean_fitnesses['train'][-1]:.1f}))...",
            f"[Total time: {self._time()}]",
        )
        print("".join(text), end="")
        # Every generation, save the progress and plot.
        # if self.generation % 10 == 0:
        self._save()
        self._plot()

    def mark_complete(self):
        self.complete = True

    def _plot(self):
        plt.figure(figsize=(6, 6))
        plt.plot(
            self.max_fitnesses["x"],
            self.max_fitnesses["train"],
            color="g",
            label="Train Max+LR",
        )
        plt.plot(
            self.max_fitnesses["x"],
            self.max_fitnesses["eval"],
            color="b",
            lw=2,
            label="Eval Max+LR",
        )
        plt.plot(
            self.mean_fitnesses["x"],
            self.mean_fitnesses["train"],
            "g--",
            alpha=0.4,
            label="Train Mean+LR",
        )
        plt.plot(
            self.mean_fitnesses["x"],
            self.mean_fitnesses["eval"],
            "b--",
            alpha=0.4,
            label="Eval Mean+LR",
        )
        plt.xlim([self.max_fitnesses["x"][0], self.max_fitnesses["x"][-1]])
        plt.ylim([0.45, 1])
        plt.yticks(np.arange(0.45, 1.01, 0.05))
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend(loc="upper left")
        plt.savefig(f"{self.save_path}_progress.pdf")
        plt.close()
        plt.figure(figsize=(6, 6))
        plt.plot(self.times)
        plt.xlabel("Generations")
        plt.ylabel("Time per iteration")
        plt.savefig(f"{self.save_path}_time.pdf")
        plt.close()

    def _time(self):
        tic = time.time()
        # Store the iteration time.
        self.times.append(tic - self._last_tic)
        self._last_tic = tic
        # Print the elapsed time.
        elapsed = int(tic - self._start_time)
        seconds = elapsed % 60
        t = "%.1fs" % seconds
        minutes = elapsed // 60
        if minutes > 0:
            t = ("%dm" % (minutes % 60)) + t
            hours = minutes // 60
            if hours > 0:
                t = ("%dh" % hours) + t
        return t

    def _save(self):
        # results = ff  # Default: store all.
        results = {
            "population": self.query_search.population,
            "max_fitnesses": self.max_fitnesses,
            "mean_fitnesses": self.mean_fitnesses,
            "times": self.times,
        }
        with open(f"{self.save_path}_generationlog.pickle", "wb") as ff:
            pickle.dump(results, ff)
