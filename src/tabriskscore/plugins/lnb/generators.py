### add generators
from abc import ABC

import torch
from reprosyn.methods import (CTGAN, DS_BAYNET, DS_INDHIST, DS_PRIVBAYES,
                              SYNTHPOP)


class Generator(ABC):
    """Base class for generators"""

    def __init__(self):
        self.trained = False

    @property
    def label(self):
        return "Unnamed Generator"

    def __str__(self):
        return self.label


class identity(Generator):
    """This generator is the identity generator: just return the input dataset."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        return dataset

    @property
    def label(self):
        return "identity"


class baynet(Generator):
    """This generator is based on BAYNET."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        baynet = DS_BAYNET(dataset=dataset, metadata=metadata, size=size, seed=seed)
        baynet.run()
        return baynet.output

    @property
    def label(self):
        return "BAYNET"


class privbayes(Generator):
    """This generator is based on privbayes."""

    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        pbayes = DS_PRIVBAYES(
            dataset=dataset,
            metadata=metadata,
            size=size,
            epsilon=self.epsilon,
            seed=seed,
        )
        pbayes.run()
        return pbayes.output

    @property
    def label(self):
        return "privbayes"


class ctgan(Generator):
    """This generator is based on CTGAN."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed, epochs=50):
        torch.manual_seed(seed)
        ctgan = CTGAN(dataset=dataset, metadata=metadata, size=size, epochs=epochs)
        ctgan.run()
        return ctgan.output

    @property
    def label(self):
        return "CTGAN"


class synthpop(Generator):
    """This generator is based on SYNTHPOP."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        spop = SYNTHPOP(dataset=dataset, metadata=metadata, size=size, seed=seed)
        spop.run()
        return spop.output

    @property
    def label(self):
        return "SYNTHPOP"


class indhist(Generator):
    """This generator is based on INDHIST."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        indhist = DS_INDHIST(dataset=dataset, metadata=metadata, size=size)
        indhist.run()
        return indhist.output

    @property
    def label(self):
        return "INDHIST"


def get_generator(name_generator: str, epsilon: float):
    """Get generator instance

    :param name_generator: generator name. Supports "identity", "BAYNET", "privbayes", "CTGAN", "SYNTHPOP", "INDHIST"
    :type name_generator: str
    :param epsilon: epsilon for training differentially private generators
    :type epsilon: float
    :return: generator instance
    :rtype: Generator
    """
    if name_generator == "identity":
        return identity()
    if name_generator == "BAYNET":
        return baynet()
    if name_generator == "privbayes":
        return privbayes(epsilon)
    if name_generator == "CTGAN":
        return ctgan()
    if name_generator == "SYNTHPOP":
        return synthpop()
    if name_generator == "INDHIST":
        return indhist()
    print("Not a valid generator.")
