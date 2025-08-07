"""Wrapper module for the C qbs library."""

# The C module to wrap.
import cqbs


class QBS_TYPE:
    """Constants: enum number for QBS types."""

    SIMPLE = 0
    DIFFIX = 1
    TABLEBUILDER = 2
    DPLAPLACE = 3


class QueryBasedSystem:
    """Running instance of a Query-Based Systems.

    This class provides a Python-friendly interface to the methods of cqbs,
    by safeguarding the QBS instance (address) and wrapping the different
    methods of cqbs."""

    def __init__(self, dataset, qbs_parameters):
        """Create a QBS instance covering a dataset. The instance is specified
        by the qbs_parameters dictionary, which follows the semantic defined
        in qbsmodule.c. Do not use this method directly unless you truly know
        what you are doing (use QBS-specific classes instead).

        INPUTS:
         - `dataset`, a list of tuples of integers of equal size.
         - `qbs_parameters`, a dictionary mapping parameter:value to describe
             the query-based system (see qbsmodule.c for specifics).
        """
        self.instance = cqbs.create_qbs(list(dataset), qbs_parameters)

    def query(self, values, conditions, budget_fractions=None):
        """Perform one or more queries with arbitrary values on this QBS.

        Each query is represented by a combination of a value tuple and a
        condition tuple, each of length num_attributes. The semantic is
          COUNT( AND_i ( attribute_i OPERATOR(condition_i) value_i ) ),
        where OPERATOR(c) is defined as:
            0  ->  no condition on this attribute;
            1  ->  ==
           -1  ->  !=
            2  ->  >
            3  ->  >=
           -2  ->  <
           -3  ->  <=

        INPUTS:
         - values, a list of tuples, each of integers and of length num_attributes;
         - conditions, a list of tuples, each of integers and of length (idem);
         - budget_fractions, a list of floats of length (idem).

         It must be that len(values) == len(conditions) [== len(budget_fractions)].

        OUTPUT:
         - The answer to each query, as an integer, in a list of length num_queries.
        """
        assert len(values) == len(
            conditions
        ), "Inputs values and conditions should have the same length."
        if budget_fractions:
            assert len(values) == len(
                budget_fractions
            ), "Inputs values and budget_fractions should have the same length."
            return cqbs.query_qbs(self.instance, values, conditions, budget_fractions)
        return cqbs.query_qbs(self.instance, values, conditions)

    def structured_query(self, users, conditions, budget_fractions=None):
        """Perform a structured query (matching users) on this QBS.

               Structured queries do not require explicit values to be
               tested in the

               condition. The semantic is, for a query
                (user, conditions):
                 COUNT( AND_i ( attribute_i OPERATOR(condition_i)
                 user(attribute_i)) )
               where OPERATOR(c) is defined the same way as in .query.

               INPUTS:
                - users, a list of integers, representing the users
                  providing values for the queries.
                - conditions, a list of tuples, each of integers and of
                  length num_attributes.
        - budget_fractions, a list of floats of same length as conditions.

               OUTPUT:
                - The answer to each query, as an integer, in a list of
                  length len(users).
        """
        if budget_fractions is not None:
            assert len(conditions) == len(
                budget_fractions
            ), "len(conditions) must be equal to len(budget_fractions)."
            return cqbs.structured_query_qbs(
                self.instance, users, conditions, budget_fractions
            )
        return cqbs.structured_query_qbs(self.instance, users, conditions)

    def __del__(self):
        """Called when the object is about to be destroyed: free the memory.

        This needs to be done manually because the QBS instance is malloc'd
        in the C code (and is thus not managed by the Python garbage collector).

        Do *not* call free_qbs on this instance manually -- this will cause
        issues when this object is __del__'d. If you wish to release memory,
        instead use the del operator on the QBS instance."""
        cqbs.free_qbs(self.instance)


class Diffix(QueryBasedSystem):
    """Implementation of a simple version of the Diffix Aspen QBS."""

    def __init__(self, dataset, seed=0):
        QueryBasedSystem.__init__(
            self, dataset, {"type": QBS_TYPE.DIFFIX, "seed": seed}
        )


class SimpleQBS(QueryBasedSystem):
    """Implementation of a simple QBS, with:
    - Bucket suppression on the exact answer, if(x<=t) -> 0.
    - Random noise addition if not bucket suppressed, + N(0, scale^2)."""

    def __init__(self, dataset, bucket_threshold=0, noise_scale=0, seed=0):
        QueryBasedSystem.__init__(
            self,
            dataset,
            {
                "type": QBS_TYPE.SIMPLE,
                "bucket_threshold": int(bucket_threshold),
                "noise_scale": float(noise_scale),
                "seed": seed,
            },
        )


class TableBuilder(QueryBasedSystem):
    """Implementation of TableBuilder with threshold and uniform noise."""

    def __init__(self, dataset, threshold=4, noise_scale=2, seed=0):
        QueryBasedSystem.__init__(
            self,
            dataset,
            {
                "type": QBS_TYPE.TABLEBUILDER,
                "threshold": threshold,
                "noise_scale": noise_scale,
                "seed": seed,
            },
        )


class DPLaplace(QueryBasedSystem):
    """Implementation of the Differentially Private Laplace mechanism."""

    def __init__(self, dataset, epsilon, seed=0):
        QueryBasedSystem.__init__(
            self,
            dataset,
            {"type": QBS_TYPE.DPLAPLACE, "epsilon": epsilon, "seed": seed},
        )

    # budget_fractions *must* be provided for this QBS.
    # We thus override the query and structured_query methods to enforce it.
    def query(self, values, conditions, budget_fractions):
        return QueryBasedSystem.query(self, values, conditions, budget_fractions)

    def structured_query(self, users, conditions, budget_fractions):
        return QueryBasedSystem.structured_query(
            self, users, conditions, budget_fractions
        )
