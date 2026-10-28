import numpy as np


def narayanas_cows(num_terms):
    """Returns Narayana's cows sequence.

    Each term in the sequence is the sum of the previous term and the term
    three places before it. The first three terms are 1, 1, 1. The ratio
    between consecutive terms approaches the supergolden ratio.

    Args:
        num_terms: Integer specifying the total number of terms in the
            sequence to return.

    Returns:
        out: Array containing the sequence of integers.

    References:
        The On-line Encyclopedia of Integer Sequences. Entry A000930.
        https://oeis.org/A000930
    """
    seq = np.zeros(num_terms, dtype=int)
    seq[:3] = 1
    for i in range(3, num_terms):
        seq[i] = seq[i - 1] + seq[i - 3]

    return seq
