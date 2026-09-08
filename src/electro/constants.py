import numpy as np


def supergolden_ratio():
    # Polynomial coefficients
    coeffs = [1, -1, 0, -1]
    roots = np.roots(coeffs)
    # Check that there is only one real root
    assert sum([1 if root.imag == 0 else 0 for root in roots]) == 1
    root = roots[0].real
    assert root > 0
    return root
