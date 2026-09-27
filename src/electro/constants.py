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


def golden_mean_2d(reverse=False):
    """Returns the 2D golden mean.

    Args:
        reverse: Boolean specifying whether to reverse the order of the
            components. True corresponds to the order used in Chan et al.

    References:
        R. W. Chan et al. (2009). Temporal stability of adaptive 3D radial MRI
        using multidimensional golden means. Magnetic Resonance in Medicine.
        https://doi.org/10.1002/mrm.21837
    """
    p = supergolden_ratio()
    if reverse:
        return np.array([1 / p ** 2, 1 / p])
    else:
        return np.array([1 / p, 1 / p ** 2])


def plastic_ratio():
    # Polynomial coefficients
    coeffs = [1, 0, -1, -1]
    roots = np.roots(coeffs)
    # Check that there is only one real root
    assert sum([1 if root.imag == 0 else 0 for root in roots]) == 1
    root = roots[0].real
    assert root > 0
    return root


def plastic_point(reverse=False):
    """Returns the 2D point corresponding to the plastic ratio.

    Args:
        reverse: Boolean specifying whether to reverse the order of the
            components.
    """
    p = plastic_ratio()
    if reverse:
        return np.array([1 / p ** 2, 1 / p])
    else:
        return np.array([1 / p, 1 / p ** 2])
