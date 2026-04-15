import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from seas.ica import rebuild
from typing import Dict, Optional, Literal


def apply_pca(
    components: Dict,
    n_components: int=10,
    solver: Literal["auto", "full", "randomized", "svd", "incremental"]="randomized",
    include_noise: bool=True,
    addback_mean: Literal["full", "lowpass", "highpass"]="full",
    lowpass_hz: float=0.1,
    highpass_hz: float=0.5,
    fps: float=7.5,
    # NEW: choose which centering to apply
    center_mode: Literal["pixel", "frame", "both", "none"] = "pixel",
    # existing: per-pixel standardization (applied after optional frame-centering)
    standardize: Literal["center", "zscore", "none"] = "center",
    batch_size: Optional[int] = None, # for IncrementalPCA
    return_qc: bool=True
):
    """
    PCA on a (optionally) artifact-cleaned rebuild of a calcium imaging recording.

    center_mode:
        "pixel" : subtract each pixel's temporal mean (column-center)  [previous default]
        "frame" : subtract the framewise spatial mean at each time (row-center)
        "both"  : do frame-centering, then pixel-centering
        "none"  : no mean removal

    Returns:
        dict with
            - 'pca_timecourse': (n_components, T)
            - 'pca_maps': (n_components, X, Y)
            - 'explained_variance_ratio': (n_components,)
            - 'used_pixels_mask': (X, Y) bool mask of analyzed pixels
            - 'params': dict of settings
            - 'qc': correlations vs framewise mean (if return_qc)
    """
    cp = components  # seas.ica.project output

    apply_filter_mean = (addback_mean != "full")
    filter_method_lookup = {
        "full":    "wavelet",            # ignored when apply_mean_filter=False
        "lowpass": "butterworth_lowpass",
        "highpass":"wavelet"
    }
    low_cutoff = highpass_hz if addback_mean == "highpass" else lowpass_hz

    movie = rebuild(
        cp,
        include_noise=include_noise,
        apply_mean_filter=apply_filter_mean,
        filter_method=filter_method_lookup[addback_mean],
        fps=fps,
        low_cutoff=low_cutoff
    )

    T, X, Y = movie.shape
    print(f"The rebuilt movie has shape: {movie.shape}")

    roimask = cp.get("roimask", None)
    if roimask is not None:
        roi_idx = (roimask.astype(bool)).ravel()
    else:
        roi_idx = np.ones(X*Y, dtype=bool)

    M = movie.reshape(T, X*Y)[:, roi_idx]  # (T, pixels_in_mask)

    # drop columns with NaNs or near-constant variance
    finite_col = np.isfinite(M).all(axis=0)
    var = M.var(axis=0)
    good_col = finite_col & (var > 1e-12)
    M = M[:, good_col]
    used_mask = np.zeros(X*Y, dtype=bool)
    used_mask[roi_idx] = good_col
    used_mask = used_mask.reshape(X, Y)

    print("Running PCA on rebuilt video\n-----------------------")

    # ---- centering / standardization ----
    if center_mode not in ("pixel", "frame", "both", "none"):
        raise ValueError("center_mode must be 'pixel','frame','both','none'.")

    # 1) optional framewise (row) centering: remove uniform mode per frame
    if center_mode in ("frame", "both"):
        M = M - M.mean(axis=1, keepdims=True)

    # 2) optional per-pixel (column) standardization
    if center_mode in ("pixel", "both") or center_mode == "none":
        if standardize == "zscore":
            M = StandardScaler(with_mean=True, with_std=True).fit_transform(M)
        elif standardize == "center":
            M = M - M.mean(axis=0, keepdims=True)
        elif standardize == "none":
            pass
        else:
            raise ValueError("Standardize must be 'center', 'zscore', or 'none'.")

    # ---- PCA fit ----
    n_comp = min(n_components, min(T, M.shape[1]) - 1)
    if solver == "incremental":
        ipca = IncrementalPCA(n_components=n_comp, batch_size=batch_size or 1024)
        TC = ipca.fit_transform(M)  # (T, n_comp)
        comps = ipca.components_    # (n_comp, P_used)
        evr = ipca.explained_variance_ratio_
    elif solver in ("randomized", "full", "auto"):
        svd_solver = "randomized" if solver == "auto" else solver
        pca = PCA(n_components=n_comp, svd_solver=svd_solver, random_state=0)
        TC = pca.fit_transform(M)
        comps = pca.components_
        evr = pca.explained_variance_ratio_
    elif solver == "svd":
        # TruncatedSVD works without centering; still fine after centering
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        TC = svd.fit_transform(M)
        comps = svd.components_
        evr = svd.explained_variance_ratio_
    else:
        raise ValueError("Solver must be 'full', 'randomized', 'auto', 'incremental', or 'svd'.")

    pca_timecourse = TC.T.astype(np.float32)  # (n_comp, T)
    # insert back to full (X*Y), filling outside with NaN, then reshape
    pca_maps = np.full((n_comp, X*Y), np.nan, dtype=np.float32)
    pca_maps[:, used_mask.ravel()] = comps.astype(np.float32)
    pca_maps = pca_maps.reshape(n_comp, X, Y)

    out = {
        "pca_timecourse": pca_timecourse,
        "pca_maps": pca_maps,
        "explained_variance_ratio": evr.astype(np.float32),
        "used_pixels_mask": used_mask,
        "params": {
            "n_components": n_comp,
            "solver": solver,
            "include_noise": include_noise,
            "addback_mean": addback_mean,
            "lowpass_hz": lowpass_hz,
            "highpass_hz": highpass_hz,
            "fps": fps,
            "center_mode": center_mode,
            "standardize": standardize
        },
    }

    if return_qc:
        mean_tc = (cp["mean"][:T] - np.mean(cp["mean"][:T])) / (np.std(cp["mean"][:T]) + 1e-8)
        tc_z = (pca_timecourse - pca_timecourse.mean(axis=1, keepdims=True)) / (pca_timecourse.std(axis=1, keepdims=True) + 1e-8)
        mean_corr = (tc_z @ mean_tc) / T  # (n_comp,)
        out["qc"] = {"mean_corr": mean_corr.astype(np.float32)}

    return out
