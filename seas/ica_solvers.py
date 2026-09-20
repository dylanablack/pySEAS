"""Optional ICA solvers with PySEAS's samples=pixels convention.

Picard uses explicit preprocessing matching sklearn FastICA's eigh whitening
and unit-variance outputs. FastICA itself remains unmodified. Regression tests
check this compatibility; it must be checked again on dependency upgrades.
"""

import warnings

import numpy as np
from scipy import linalg
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_array


def solver_settings(solver, tol=None):
    if solver not in ('fastica', 'picard-o'):
        raise ValueError("solver must be 'fastica' or 'picard-o'")
    if tol is None:
        tol = 1e-4 if solver == 'fastica' else 1e-7
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be positive and finite')
    if solver == 'picard-o':
        try:
            import picard
        except ImportError as error:
            raise ImportError("solver='picard-o' requires python-picard; "
                              "install seas[picard] or python-picard==0.8.2") from error
        version = picard.__version__
    else:
        from sklearn import __version__
        version = __version__
    return {'solver': solver, 'solver_version': version, 'tol': float(tol)}


def make_solver(solver, n_components, max_iter, tol, w_init=None):
    if solver == 'fastica':
        return FastICA(n_components=n_components, max_iter=max_iter,
                       tol=tol, random_state=1000, w_init=w_init,
                       whiten_solver='eigh')
    if solver != 'picard-o':
        raise ValueError("solver must be 'fastica' or 'picard-o'")
    return PicardO(n_components, max_iter, tol, w_init=w_init)


class PicardO:
    """Minimal fit_transform/mixing_ adapter; not a general sklearn estimator."""

    def __init__(self, n_components, max_iter, tol, w_init=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init

    def fit_transform(self, X):
        from picard import picard

        X = check_array(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError('max_iter must be a positive integer')
        if not isinstance(self.n_components, (int, np.integer)):
            raise ValueError('n_components must be a positive integer')
        n_samples, n_features = X.shape
        n = min(self.n_components, n_samples, n_features)
        if n < 1:
            raise ValueError('n_components must be positive')
        if n != self.n_components:
            warnings.warn(f'n_components is too large; using {n}')

        # Match FastICA's centering, eigenvector signs and variance convention.
        centered = X.copy(order='K').T
        self.mean_ = centered.mean(axis=1)
        centered -= self.mean_[:, None]
        values, vectors = linalg.eigh(centered @ X)
        floor = 10 * np.finfo(values.dtype).eps
        if np.any(values < floor):
            warnings.warn('Small eigenvalues encountered in ICA eigh whitening.')
        values = np.sqrt(np.maximum(values, floor))[::-1]
        vectors = vectors[:, ::-1]
        vectors *= np.sign(vectors[0])
        self.whitening_ = (vectors / values).T[:n]
        white = (self.whitening_ @ centered) * np.sqrt(n_samples)
        del values, vectors

        if self.w_init is None:
            initial = np.random.RandomState(1000).normal(size=(n, n)).astype(white.dtype)
        else:
            initial = np.asarray(self.w_init)
        if initial.shape != (n, n):
            raise ValueError('w_init must match n_components')
        values, vectors = linalg.eigh(initial @ initial.T)
        values = np.maximum(values, np.finfo(initial.dtype).tiny)
        initial = (vectors / np.sqrt(values)) @ vectors.T @ initial

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, weights, _, iteration_index = picard(
                white, whiten=False, centering=False, ortho=True, extended=True,
                fun='tanh', w_init=initial, max_iter=self.max_iter,
                tol=self.tol, return_n_iter=True,
            )
        # python-picard 0.8.2 reports the zero-based final loop index.
        self.n_iter_ = int(iteration_index) + 1
        for warning in caught:
            category = (ConvergenceWarning if 'Picard did not converge' in str(warning.message)
                        else warning.category)
            warnings.warn(str(warning.message), category, stacklevel=2)

        del white
        sources = np.linalg.multi_dot([weights, self.whitening_, centered]).T
        scale = sources.std(axis=0, keepdims=True)
        if np.any(scale == 0) or not np.isfinite(scale).all():
            raise ValueError('Degenerate Picard sources')
        sources /= scale
        weights /= scale.T
        self.components_ = weights @ self.whitening_
        self.mixing_ = linalg.pinv(self.components_, check_finite=False)
        return sources
