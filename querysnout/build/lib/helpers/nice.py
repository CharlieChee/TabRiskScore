"""nice.py: Nice display of query-based solutions."""

import collections

import numpy as np


class COLORS:
    # HEADER = '\033[95m'
    BOLD = "\033[1m"
    ENDC = "\033[0m"

    def f(c):
        code = str(c)  # i * 16 + j)
        return "\u001b[38;5;" + code + "m"

    # Diffix says trans rights.
    POS = f(123)
    NEG = f(211)
    AND = f(7)
    HEADER = f(254) + BOLD


class JUPYTER_COLORS:
    BOLD = "\033[1m"
    ENDC = "\033[0m"

    def f(c):
        code = str(c)  # i * 16 + j)
        return "\u001b[38;5;" + code + "m"

    # Diffix says trans rights.
    POS = f(123)
    NEG = f(211)
    AND = f(244)
    HEADER = AND + BOLD


def isolate_neq_queries(solution):
    """
    (internal) Isolate all pairs of queries that are part of a Noise-Exploitation Attack.
    This returns:
        1. List of non-neq queries.
        2. List of pairs of neq queries and the indices of the flipped entries in other queries.
           (formally, L = [(neq_i, (i1, i2, ...)), ...])
    """
    n_queries = len(solution)
    non_neq_queries = list(solution)
    neq_pairs = []
    # Iterate over queries to find "leading" queries (q in the pair (q,q')).
    for i, query1 in enumerate(solution):
        # Leading queries have no -1 in quasi-identifiers, and condition on the sensitive attribute.
        if any(cond == -1 for cond in query1[:-1]) or query1[-1] == 0:
            continue
        # This is a leading query! Find all queries that differ only by one condition, replacing a
        #  0 (no condition) by a -1 (difference condition).
        pairs_for_this_query = []
        for query2 in solution[:i] + solution[i + 1 :]:  # skip query1
            diverging_index = None
            compatible = True
            for idx, (x, y) in enumerate(zip(query1, query2)):
                # This is the only acceptable difference.
                if x == 0 and y == -1:
                    # There are > 1 differences! Not acceptable.
                    if diverging_index is not None:
                        compatible = False
                        break
                    # Otherwise, this is the index where things change.
                    diverging_index = idx
                # Any other difference is not ok.
                elif x != y:
                    compatible = False
                    break
            if compatible:
                pairs_for_this_query.append((query2, diverging_index))
                # Remove the corresponding queries from non_neq_queries.
                if query1 in non_neq_queries:
                    non_neq_queries.remove(query1)
                if query2 in non_neq_queries:
                    non_neq_queries.remove(query2)
        # We have now found all pairs for this query.
        if pairs_for_this_query:
            neq_pairs.append((query1, pairs_for_this_query))
    return non_neq_queries, neq_pairs


def _display_query_cli(
    query, bold_indices=[], multiplicity=1, COLORS=COLORS, last="ignored"
):
    AND = "%s ^ %s" % (COLORS.AND, COLORS.ENDC)
    for i, x in enumerate(query[:-1]):
        color = (
            (COLORS.BOLD if i in bold_indices else "") + COLORS.NEG
            if x == -1
            else COLORS.POS
        )
        text = ""
        if x == 1:
            text = "a%d==v%d" % (i + 1, i + 1)
        elif x == -1:
            text = "a%d!=v%d" % (i + 1, i + 1)
        else:
            text = " " * len("a%d==v%s" % (i + 1, i + 1))
        print(color + text + COLORS.ENDC, end=AND if i != (len(query) - 2) else "")
    # Add the s=0 or s=1 at the end.
    c = {
        1: AND + COLORS.POS + "s=1" + COLORS.ENDC,
        -1: AND + COLORS.NEG + "s=0" + COLORS.ENDC,
        0: " " * 6,
    }
    print(c[query[-1]], end="")
    # Finally, add the multiplicity (if >1).
    if multiplicity > 1:
        print(
            "  %s(x%s)%s" % (COLORS.BOLD + COLORS.HEADER, multiplicity, COLORS.ENDC),
            end="",
        )
    print()  # Completes the line.


LATEX_PREAMBLE = """\\begin{framed}
\\[
{\\customsize
\\begin{array}{lll}"""


def _latex_preamble(num_attributes):
    return """\\begin{framed}
\\noindent\\textbf{Example of solution found against YYY in the AUXILIARY scenario.}
\\[
{\\customsize
\\begin{array}{%s}""" % (
        "l" * (num_attributes + 2)
    )


def _display_query_latex(
    query, bold_indices=[], multiplicity=1, COLORS=COLORS, last=False
):
    AND = " & \\land~"
    texts = []
    for i, x in enumerate(query[:-1]):
        if x == 1:
            text = "\\textcolor{pos}{a_{%d} = v_{%d}}" % (i + 1, i + 1)
        elif x == -1:
            text = "\\textcolor{neg}{a_{%d} \\neq v_{%d}}" % (i + 1, i + 1)
        else:
            text = ""  # ' ' * len('a%d==v%s' % (i+1,i+1))
        texts.append(text)
    # Find the first nonzero index.
    nonzeros = [i for i, t in enumerate(texts) if t != 0]
    if not nonzeros:
        # Empty query! Nothing to show.
        print("\\\\")
        return
    idx = min(nonzeros)
    # Pad the first "count_zero_idx" entries with & column skip.
    print(" & " * idx, end="")
    print(AND.join(texts), end="")
    # Add the s=0 or s=1 at the end.
    c = {
        1: AND + "\\textcolor{pos}{s=1}",
        -1: AND + "\\textcolor{neg}{s=0}",
        0: "",
    }
    print(c[query[-1]], end=" & ")
    # Finally, add the multiplicity (if >1).
    if multiplicity > 1:
        print(
            "\\mathbf{(\\times %d)}" % multiplicity,
            end="",
        )
    if last:
        print("\\vspace{\\customspace}", end="")
    print("\\\\")  # Completes the line.


LATEX_POSTAMBLE = """\\end{array}}
\\]
\\end{framed}
"""


def display_solution(
    solution,
    isolate_neq=True,
    jupyter=False,
    silent=False,
    latex=False,
    return_indices=False,
):
    """Display a +-1 solution on a command-line interface.

    INPUT:
        - solution: list of tuples, where each tuple represents a query,
                with k+1 entries in {-1, 0, +1}.
        - isolate_neq: if True, this will isolate the noise-exploitation attack queries.
        - jupyter: special display for Jupyter notebooks.
        - silent: whether to print anything or just return the result.

    OUTPUT: prints a nice display on the command line, and returns queries in a sorted order.
    """
    DISPLAY_COLORS = JUPYTER_COLORS if jupyter else COLORS
    if silent:
        print_query = printf = lambda *args, **kwargs: None
    elif not latex:
        printf = print
        print_query = _display_query_cli
    else:
        printf = lambda *a, **k: None
        print_query = _display_query_latex
        # Do the LaTeX preamble as well.
        # print(LATEX_PREAMBLE)
        print(_latex_preamble(len(solution[0])))
    # Remove the duplicates first.
    query_multiplicity = collections.defaultdict(int)
    query_to_indices = collections.defaultdict(list)
    for i, q in enumerate(solution):
        query_multiplicity[q] += 1
        query_to_indices[q].append(i)
    solution = list(query_multiplicity.keys())
    # Sort the solutions alphabetically.
    # Add the number of queries as first entry to sort by #nonzero entries.
    solution = sorted(
        [(sum([1 for x in query if x != 0]),) + query for query in solution]
    )
    solution = [s[1:] for s in solution]  # And remove that entry.
    # If need to isolate the noise-exploitation attack, replace solution w/ non-neq solutions.
    pairs_neq = []
    if isolate_neq:
        solution, pairs_neq = isolate_neq_queries(solution)
    sorted_solution = []
    q_idx = 1
    diff_query_count = 0
    if pairs_neq:
        printf(
            "%s[Difference queries]%s" % (DISPLAY_COLORS.HEADER, DISPLAY_COLORS.ENDC)
        )
        if latex:
            print(" \\multicolumn{3}{l}{\\textbf{Difference queries}} \\\\")
            # print(" & \\textbf{Difference Queries} \\\\")
        for idx_pair, (query1, all_query2_for_query1) in enumerate(pairs_neq):
            # First, display the leading query.
            printf(
                " %s[%d]%s " % (DISPLAY_COLORS.HEADER, q_idx, DISPLAY_COLORS.ENDC),
                end="",
            )
            if latex:
                # q_idx_max = q_idx + len(all_query2_for_query1)
                print(f" ({q_idx}) & ", end="")
            print_query(query1, [], query_multiplicity[query1], DISPLAY_COLORS)
            sorted_solution += [query1]
            diff_query_count += query_multiplicity[query1]
            q_idx += 1
            # Second, individually display following queries.
            for q2q1_pair, (query2, index) in enumerate(all_query2_for_query1):
                printf(
                    " %s[%d]%s " % (DISPLAY_COLORS.HEADER, q_idx, DISPLAY_COLORS.ENDC),
                    end="",
                )
                if latex:
                    print(f" ({q_idx}) & ", end="")
                print_query(
                    query2,
                    [index],
                    query_multiplicity[query2],
                    DISPLAY_COLORS,
                    last=idx_pair == len(pairs_neq) - 1
                    and q2q1_pair == len(all_query2_for_query1) - 1,
                )
                q_idx += 1
                sorted_solution += [query2]
                diff_query_count += query_multiplicity[query2]
            printf()
    if solution:
        printf(
            "%s[%sueries]%s"
            % (
                DISPLAY_COLORS.HEADER,
                "Other q" if pairs_neq else "Q",
                DISPLAY_COLORS.ENDC,
            )
        )
        if latex:
            print(
                " \\multicolumn{3}{l}{\\textbf{%sueries}} \\\\"
                % ("Other q" if pairs_neq else "Q",)
            )
        for query in solution:
            printf(
                " %s[%d]%s " % (DISPLAY_COLORS.HEADER, q_idx, DISPLAY_COLORS.ENDC),
                end="",
            )
            if latex:
                print(f" ({q_idx}) & ", end="")
            print_query(query, [], query_multiplicity[query], DISPLAY_COLORS)
            q_idx += 1
            sorted_solution.append(query)
    if latex:
        print(LATEX_POSTAMBLE)
    if return_indices:
        # Return the sorted solution, as well as the indices of the sorted solution in
        # the original array (such that sorted_solution ~= solution[indices] (but without
        # multiplicities, so not exactly)), and the cutoff between difference queries and the rest.
        indices = [query_to_indices[q] for q in sorted_solution]
        return sorted_solution, np.concatenate(indices), diff_query_count
    else:
        # Simply return the sorted solution.
        return sorted_solution


if __name__ == "__main__":
    solution = [
        (-1, -1, 0, 0, -1, 1),
        (-1, 0, 1, 0, -1, 1),
        (-1, 1, 1, -1, 1, -1),
        (-1, 1, 1, 1, 1, 1),  # Q1'
        (0, 0, -1, 1, 1, 0),
        (0, 1, 0, -1, 0, -1),
        (0, 1, 1, 0, -1, 1),
        (0, 1, 1, 1, 1, 1),  # Q1
        (1, -1, 1, -1, -1, 0),
        (1, 0, 0, -1, -1, 1),
        (1, 0, 1, 1, 1, -1),
        (1, 0, 1, 1, 1, -1),  # Duplicate.
        (1, 1, -1, -1, 1, 0),
        (1, 1, -1, 1, 1, 1),  # Q2'
        (1, 1, 0, 1, 1, -1),
        (1, 1, 0, 1, 1, 1),  # Q2
        (1, 1, 1, -1, 0, -1),
        (1, 1, 1, -1, 1, 1),  # Q3'
        (1, 1, 1, 1, -1, 1),
        (1, 1, 1, 0, 1, -1),
        (1, 1, 1, 0, 1, 1),  # Q3
        (1, 1, 0, 0, 0, -1),  # Q4
        (1, 1, -1, 0, 0, -1),  # Q4' (1)
        (1, 1, 0, -1, 0, -1),  # Q4' (2)
    ]

    q1 = display_solution(solution, isolate_neq=True)

    print()
    q2 = display_solution(solution, isolate_neq=True, silent=True)

    print(
        "Silent version is correct!"
        if q1 == q2
        else "Silent and non-silent diverged (bad)."
    )
