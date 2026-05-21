import warnings
warnings.filterwarnings('ignore', message='.*deprecated.*')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 10)
pd.set_option('expand_frame_repr', False)
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.linalg import eigh, lstsq, norm
from scipy import stats, optimize
from scipy.optimize import nnls
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

# Optional: use LedoitWolf if scikit-learn is available; otherwise fallback to ridge shrinkage
try:
    from sklearn.covariance import LedoitWolf
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

import mylib.plotting.basic as bsc

## NECESSARY FUNCTIONS
# -----------------------------------------------------------------------------
# Robust scaling & helpers
# -----------------------------------------------------------------------------
def robust_scale_numeric(df_num: pd.DataFrame) -> pd.DataFrame:
    """Robustly scale numeric columns using median and MAD (scaled to ~SD)."""
    med = df_num.median(axis=0)
    mad = (df_num - med).abs().median(axis=0)
    # Consistency constant for normal distributions
    mad_scaled = mad * 1.4826
    mad_scaled = mad_scaled.replace(0, 1.0)  # avoid division by zero
    z = (df_num - med) / mad_scaled
    return z

def mode_or_nan(series: pd.Series):
    """Return statistical mode; if tie or empty, return NaN."""
    cnt = series.dropna()
    if cnt.empty:
        return np.nan
    mc = Counter(cnt)
    top = mc.most_common()
    if len(top) == 0:
        return np.nan
    # If multiple modes, pick the first deterministically (sorted by value then count)
    maxc = top[0][1]
    candidates = sorted([v for v, c in top if c == maxc])
    return candidates[0]

# -----------------------------------------------------------------------------
# Mahalanobis distance (robust + shrinkage)
# -----------------------------------------------------------------------------
def mahalanobis_distance_matrix(
    X: pd.DataFrame,
    robust: bool = True,
    shrinkage: str = "lw",    # 'lw' (LedoitWolf) | 'ridge' | 'none'
    ridge_alpha: float = 0.1,
    return_cov: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute an NxN Mahalanobis distance matrix between rows of X.
    - robust: robust scaling via median/MAD before covariance estimation
    - shrinkage: 'lw' for LedoitWolf (if available), else 'ridge' or 'none'
    - ridge_alpha: shrinkage weight toward identity if using 'ridge'
    """
    # Use only numeric cols for Mahalanobis
    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        raise ValueError("Mahalanobis distance requires numeric features.")

    Z = robust_scale_numeric(X_num) if robust else (X_num - X_num.mean()) / X_num.std(ddof=0).replace(0, 1.0)
    Z = Z.to_numpy(dtype=float)
    n, p = Z.shape

    # Covariance with shrinkage
    if shrinkage == "lw" and _HAVE_SKLEARN:
        lw = LedoitWolf(store_precision=True).fit(Z)
        Sigma = lw.covariance_
        Sigma_inv = lw.precision_
    else:
        # Sample covariance
        S = np.cov(Z, rowvar=False, ddof=1)
        # Ridge shrinkage to identity if requested or if sklearn unavailable
        if shrinkage == "ridge":
            trS_over_p = np.trace(S) / p
            Sigma = (1 - ridge_alpha) * S + ridge_alpha * trS_over_p * np.eye(p)
        elif shrinkage == "none":
            Sigma = S
        else:
            # Fallback: mild ridge if 'lw' requested but sklearn unavailable
            trS_over_p = np.trace(S) / p
            Sigma = 0.9 * S + 0.1 * trS_over_p * np.eye(p)
        # Invert robustly
        Sigma_inv = np.linalg.pinv(Sigma)

    # Efficient pairwise Mahalanobis via quadratic form expansion
    AX = Z @ Sigma_inv
    diag_q = np.einsum('ij,ij->i', AX, Z)  # row-wise quadratic term
    D2 = diag_q[:, None] + diag_q[None, :] - 2.0 * (AX @ Z.T)
    D2 = np.maximum(D2, 0.0)  # numerical stability
    D = np.sqrt(D2)

    return (D, Sigma) if return_cov else (D, None)

# -----------------------------------------------------------------------------
# Spearman (1 - rho) distance
# -----------------------------------------------------------------------------
def spearman_distance_matrix(X: pd.DataFrame) -> np.ndarray:
    """
    Compute NxN distance matrix using Spearman correlation between row-vectors (across features):
    distance = 1 - Spearman rho.
    """
    # Use numeric columns; if none, attempt to convert categories to codes
    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        # Fallback: try to code non-numeric as category codes
        X_num = X.apply(lambda s: pd.Categorical(s).codes if s.dtype == 'O' else s).astype(float)

    A = X_num.to_numpy(dtype=float)
    n, p = A.shape
    # Rank within each row (across features)
    ranks = np.apply_along_axis(stats.rankdata, 1, A)
    # Z-score ranks to get Pearson = Spearman
    ranks = (ranks - ranks.mean(axis=1, keepdims=True)) / ranks.std(axis=1, ddof=0, keepdims=True)
    # Corr = (ranks @ ranks.T) / (p - 1), because each row is standardized
    corr = (ranks @ ranks.T) / (p - 1)
    corr = np.clip(corr, -1.0, 1.0)
    D = 1.0 - corr
    np.fill_diagonal(D, 0.0)
    # Correct small negative distances after clipping
    D = np.maximum(D, 0.0)
    return D

# -----------------------------------------------------------------------------
# Gower distance (mixed types)
# -----------------------------------------------------------------------------
def gower_distance_matrix(
    X: pd.DataFrame,
    feature_info: Optional[Dict[str, Dict[str, Any]]] = None,
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Compute NxN Gower distance for mixed data.
    feature_info: optional dict {col: {'type': 'numeric'|'categorical'|'ordinal', 'order': [...]} }
        - If 'ordinal', provide an 'order' list (low->high). Values are mapped to 0..1 range.
    weights: optional per-feature weights {col: weight}; default = 1.0 for all.

    Missing values are supported (pairwise feature availability).
    """
    df = X.copy()
    n = df.shape[0]
    cols = list(df.columns)

    # Infer basic types if not provided
    finfo = feature_info.copy() if feature_info else {}
    for c in cols:
        if c not in finfo:
            if pd.api.types.is_numeric_dtype(df[c]):
                finfo[c] = {'type': 'numeric'}
            elif pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == object:
                finfo[c] = {'type': 'categorical'}
            else:
                finfo[c] = {'type': 'categorical'}

    # Prepare per-feature transformed arrays and ranges
    W = np.ones(len(cols), dtype=float)
    if weights:
        for i, c in enumerate(cols):
            if c in weights:
                W[i] = float(weights[c])

    # Transform columns according to type
    transformed = []
    ranges = []
    s_is_numeric = []
    for c in cols:
        info = finfo[c]
        t = info.get('type', 'numeric')
        s = df[c]

        if t == 'numeric':
            arr = s.astype(float).to_numpy()
            x_min, x_max = np.nanmin(arr), np.nanmax(arr)
            rng = x_max - x_min if np.isfinite(x_max - x_min) and (x_max - x_min) > 0 else 1.0
            # normalized to [0,1] for numeric
            arr_norm = (arr - x_min) / rng
            transformed.append(arr_norm)
            ranges.append(1.0)  # already normalized
            s_is_numeric.append(True)

        elif t == 'ordinal':
            order = info.get('order', None)
            if order is None:
                # If no explicit order, fall back to categorical
                cats = pd.Categorical(s)
                arr = cats.codes.astype(float)
                K = len(cats.categories)
                rng = max(K - 1, 1)
                arr_norm = arr / rng
            else:
                mapping = {v: i for i, v in enumerate(order)}
                arr = s.map(mapping).astype(float).to_numpy()
                K = len(order)
                rng = max(K - 1, 1)
                arr_norm = arr / rng
            transformed.append(arr_norm)
            ranges.append(1.0)
            s_is_numeric.append(True)

        else:  # categorical
            # Keep original values; we will compare equality later
            transformed.append(s.to_numpy())
            ranges.append(1.0)
            s_is_numeric.append(False)

    T = np.vstack(transformed).T  # shape (n, p)
    Wv = W  # (p,)

    # Pairwise Gower
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi = T[i]
        for j in range(i + 1, n):
            xj = T[j]
            sij = 0.0  # sum of weights with non-missing for this pair
            dij = 0.0  # weighted sum of per-feature distances

            for k in range(len(cols)):
                wi = Wv[k]
                vi = xi[k]
                vj = xj[k]

                # Missingness handling
                if pd.isna(vi) or pd.isna(vj):
                    continue

                if s_is_numeric[k]:
                    # numeric/ordinal normalized in [0,1]
                    d = abs(float(vi) - float(vj))  # already normalized by range
                else:
                    # categorical: 0 if equal, else 1
                    d = 0.0 if vi == vj else 1.0

                dij += wi * d
                sij += wi

            if sij == 0.0:
                d_pair = np.nan  # no overlapping features
            else:
                d_pair = dij / sij

            D[i, j] = D[j, i] = d_pair if np.isfinite(d_pair) else 0.0

    np.fill_diagonal(D, 0.0)
    return D

# -----------------------------------------------------------------------------
def compute_centroid(
    X: pd.DataFrame,
    feature_info: Optional[Dict[str, Dict[str, Any]]] = None
) -> pd.Series:
    """Compute a robust centroid (median for numeric; mode for categorical)."""
    centroid = {}
    finfo = feature_info or {}
    for c in X.columns:
        info = finfo.get(c, {})
        if pd.api.types.is_numeric_dtype(X[c]) or info.get('type') in ('numeric', 'ordinal'):
            centroid[c] = X[c].median()
        else:
            centroid[c] = mode_or_nan(X[c])
    return pd.Series(centroid, index=X.columns)

def distance_to_centroid(
    X: pd.DataFrame,
    metric: str,
    feature_info: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs
) -> np.ndarray:
    """
    Distance of each row to the dataset centroid under a given metric.
    metric: 'mahalanobis' | 'spearman' | 'gower'
    kwargs passed to underlying distance functions (e.g., shrinkage, etc.)
    """
    c = compute_centroid(X, feature_info=feature_info)
    # Build a 1-row DataFrame for the centroid
    C = pd.DataFrame([c], columns=X.columns)

    if metric == 'mahalanobis':
        # Concatenate centroid with X, then take distances from centroid row to others
        tmp = pd.concat([C, X], ignore_index=True)
        D, _ = mahalanobis_distance_matrix(tmp, **{k: v for k, v in kwargs.items() if k in ['robust', 'shrinkage', 'ridge_alpha']})
        return D[0, 1:]  # distances from centroid (row 0) to each subject

    elif metric == 'spearman':
        tmp = pd.concat([C, X], ignore_index=True)
        D = spearman_distance_matrix(tmp)
        return D[0, 1:]

    elif metric == 'gower':
        tmp = pd.concat([C, X], ignore_index=True)
        D = gower_distance_matrix(tmp, feature_info=feature_info, weights=kwargs.get('weights'))
        return D[0, 1:]

    else:
        raise ValueError("Unknown metric. Choose from 'mahalanobis', 'spearman', 'gower'.")

# -----------------------------------------------------------------------------
def average_dissimilarity_from_D(D: np.ndarray) -> np.ndarray:
    """For each node i, compute mean distance to all *other* nodes."""
    n = D.shape[0]
    # avoid self-distance (0) in the mean
    return (D.sum(axis=1) - np.diag(D)) / (n - 1)

# -----------------------------------------------------------------------------
# Bootstrap (subsampling) stability
# -----------------------------------------------------------------------------
def bootstrap_individual_metrics(
    X: pd.DataFrame,
    metric: str = 'mahalanobis',
    B: int = 500,
    subsample_frac: float = 0.8,
    random_state: Optional[int] = 42,
    feature_info: Optional[Dict[str, Dict[str, Any]]] = None,
    weights: Optional[Dict[str, float]] = None,
    # Mahalanobis-specific options
    robust: bool = True,
    shrinkage: str = 'lw',
    ridge_alpha: float = 0.1,
    ci: Tuple[float, float] = (2.5, 97.5)
) -> pd.DataFrame:
    """
    Subsampling bootstrap (without replacement) to estimate stability of:
    - distinctiveness: distance to centroid
    - average dissimilarity: mean pairwise distance to others
    Returns per-subject mean and percentile CI over resamples where the subject is included.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    m = max(2, int(np.floor(subsample_frac * n)))  # at least size 2

    # Store lists of values per subject
    Dst_vals = [[] for _ in range(n)]
    Avg_vals = [[] for _ in range(n)]

    for b in range(B):
        idx = rng.choice(n, size=m, replace=False)
        Xb = X.loc[idx].reset_index(drop=True)

        # Distance matrix for the subset
        if metric == 'mahalanobis':
            Db, _ = mahalanobis_distance_matrix(Xb, robust=robust, shrinkage=shrinkage, ridge_alpha=ridge_alpha)
        elif metric == 'spearman':
            Db = spearman_distance_matrix(Xb)
        elif metric == 'gower':
            Db = gower_distance_matrix(Xb, feature_info=feature_info, weights=weights)
        else:
            raise ValueError("metric must be 'mahalanobis', 'spearman', or 'gower'.")

        # Distinctiveness to subset centroid (metric-specific)
        D_cent = distance_to_centroid(
            Xb,
            metric=metric,
            feature_info=feature_info,
            robust=robust,
            shrinkage=shrinkage,
            ridge_alpha=ridge_alpha,
            weights=weights
        )
        # Average dissimilarity within subset
        avg_dis = average_dissimilarity_from_D(Db)

        # Deposit values back to original indices
        for pos, orig_i in enumerate(idx):
            Dst_vals[orig_i].append(float(D_cent[pos]))
            Avg_vals[orig_i].append(float(avg_dis[pos]))

    def summarize(vs: list, lo: float, hi: float) -> Tuple[float, float, float, int]:
        if len(vs) == 0:
            return (np.nan, np.nan, np.nan, 0)
        arr = np.asarray(vs, dtype=float)
        return (
            float(np.nanmean(arr)),
            float(np.nanpercentile(arr, lo)),
            float(np.nanpercentile(arr, hi)),
            int(arr.size)
        )

    rows = []
    for i in range(n):
        m1, l1, h1, cnt1 = summarize(Dst_vals[i], ci[0], ci[1])
        m2, l2, h2, cnt2 = summarize(Avg_vals[i], ci[0], ci[1])
        rows.append({
            'subject_index': i,
            'distinctiveness_mean': m1,
            'distinctiveness_ci_low': l1,
            'distinctiveness_ci_high': h1,
            'average_dissim_mean': m2,
            'average_dissim_ci_low': l2,
            'average_dissim_ci_high': h2,
            'n_inclusions': cnt1  # equals cnt2
        })

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# ICC (Shrout & Fleiss)
# -----------------------------------------------------------------------------
def compute_icc(
    df_long: pd.DataFrame,
    subject_col: str,
    rater_col: str,
    score_col: str,
    icc: str = 'ICC3',       # 'ICC1' | 'ICC2' | 'ICC3'
    average: str = 'single'  # 'single' | 'average'
) -> Dict[str, Any]:
    """
    Compute ICC following Shrout & Fleiss (1979) and McGraw & Wong (1996) style formulas.
    - ICC1: one-way random effects (absolute agreement)
    - ICC2: two-way random effects (absolute agreement) ~ ICC(A,1)
    - ICC3: two-way mixed effects (consistency) ~ ICC(C,1)

    Assumes a balanced design: each subject rated by all raters once.
    Returns a dict with MS terms and ICC value.
    """
    data = df_long[[subject_col, rater_col, score_col]].copy()
    # Balance check
    n_per_subject = data.groupby(subject_col)[rater_col].nunique()
    n_per_rater = data.groupby(rater_col)[subject_col].nunique()
    if n_per_subject.nunique() != 1 or n_per_rater.nunique() != 1:
        raise ValueError("Unbalanced design detected. Each subject must be scored by all raters exactly once.")

    subjects = data[subject_col].unique()
    raters = data[rater_col].unique()
    n = len(subjects)
    k = len(raters)

    # Pivot to matrix (n x k)
    M = data.pivot(index=subject_col, columns=rater_col, values=score_col).loc[subjects, raters].to_numpy(dtype=float)
    grand = M.mean()
    mean_subject = M.mean(axis=1)
    mean_rater = M.mean(axis=0)

    # Sum of squares
    SS_total = ((M - grand) ** 2).sum()
    SS_subject = k * ((mean_subject - grand) ** 2).sum()
    SS_rater = n * ((mean_rater - grand) ** 2).sum()
    SS_error = SS_total - SS_subject - SS_rater

    # Mean squares
    MSR = SS_subject / (n - 1)  # subjects (rows)
    MSC = SS_rater / (k - 1)  # raters (columns)
    MSE = SS_error / ((n - 1) * (k - 1))

    # ICC formulas
    icc = icc.upper()
    average = average.lower()

    if icc == 'ICC1':
        # One-way random, absolute agreement
        icc_single = (MSR - MSE) / (MSR + (k - 1) * MSE)
        icc_average = (MSR - MSE) / MSR
    elif icc == 'ICC2':
        # Two-way random, absolute agreement (A,1)
        icc_single = (MSR - MSE) / (MSR + (k - 1) * MSE + (k * (MSC - MSE) / n))
        icc_average = (MSR - MSE) / (MSR + (MSC - MSE) / n)
    elif icc == 'ICC3':
        # Two-way mixed, consistency (C,1)
        icc_single = (MSR - MSE) / (MSR + (k - 1) * MSE)
        icc_average = (MSR - MSE) / MSR
    else:
        raise ValueError("icc must be one of {'ICC1','ICC2','ICC3'}.")

    val = icc_single if average == 'single' else icc_average
    return {
        'icc_type': icc,
        'unit': average,
        'ICC': float(val),
        'MSR': float(MSR), 'MSC': float(MSC), 'MSE': float(MSE),
        'n_subjects': int(n), 'n_raters': int(k)
    }


# -----------------------------------------------------------------------------
def compute_all_distance_matrices(
        X: pd.DataFrame,
        feature_info: Optional[Dict[str, Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None,
        robust_maha: bool = True,
        shrinkage: str = 'lw',
        ridge_alpha: float = 0.1
) -> Dict[str, np.ndarray]:
    """Return a dict with 'mahalanobis', 'spearman', and 'gower' distance matrices."""
    D_maha, _ = mahalanobis_distance_matrix(X, robust=robust_maha, shrinkage=shrinkage, ridge_alpha=ridge_alpha)
    D_spe = spearman_distance_matrix(X)
    D_gow = gower_distance_matrix(X, feature_info=feature_info, weights=weights)
    return {'mahalanobis': D_maha, 'spearman': D_spe, 'gower': D_gow}


# -----------------------------------------------------------------------------
# below are functions for modeling age effect on inter-indiv diff
# -----------------------------------------------------------------------------
def triu_vectorize(M: np.ndarray, k: int = 1) -> np.ndarray:
    """Vectorize the upper triangular part (k=1 excludes diagonal)."""
    i, j = np.triu_indices_from(M, k=k)
    return M[i, j]


def unvec_ut(v: np.ndarray, n: int, fill_diag=np.nan) -> np.ndarray:
    """Rebuild an NxN symmetric matrix from its upper-tri vector."""
    M = np.full((n, n), fill_diag, dtype=float)
    iu, ju = np.triu_indices(n, 1)
    M[iu, ju] = v
    M[ju, iu] = v
    return M


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR. Returns: rejected mask, critical q."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    thresh = alpha * ranks / n
    p_sorted = p[order]
    is_sig_sorted = p_sorted <= thresh
    if not np.any(is_sig_sorted):
        return np.zeros_like(p, dtype=bool), 0.0
    kmax = np.max(np.where(is_sig_sorted)[0]) + 1
    crit = thresh[kmax - 1]
    # reject those with p <= crit
    reject = p <= crit
    return reject, crit


def zscore(vec: np.ndarray) -> np.ndarray:
    """Z-score a 1D vector (safe for constant vectors)."""
    v = np.asarray(vec, float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    if s == 0 or not np.isfinite(s):
        return np.zeros_like(v)
    return (v - m) / s


def _rank_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Spearman via ranking + Pearson."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    return np.corrcoef(rx, ry)[0, 1]


def permute_labels(n, strata: Optional[np.ndarray], rng):
    """Return a permutation of 0..n-1 (within-strata if given)."""
    if strata is None:
        return rng.permutation(n)
    out = np.arange(n)
    g = np.asarray(strata)
    for lab in np.unique(g):
        idx = np.where(g == lab)[0]
        out[idx] = out[idx][rng.permutation(len(idx))]
    return out


# -----------------------------------------------------------------------------
# Similarity from distance
# -----------------------------------------------------------------------------
def distance_to_similarity(
        D: np.ndarray,
        method: str = "rbf",
        rbf_sigma: Optional[float] = None,
        eps: float = 1e-9
) -> np.ndarray:
    """
    Convert a distance matrix to a similarity matrix using a monotonic transform.
    Methods:
    - 'rbf': S = exp(-(D/sigma)^2) with sigma = median(upper(D)) if not provided
    - 'maxminus': S = (max(D) - D) / max(D)
    - 'inv': S = 1 / (1 + D)
    - 'neg': S = -D (monotonic; suitable when using rank-based stats)
    """
    D = np.asarray(D, float)
    assert D.ndim == 2 and D.shape[0] == D.shape[1], "D must be square."
    if method == "rbf":
        if rbf_sigma is None:
            du = triu_vectorize(D, k=1)
            rbf_sigma = np.nanmedian(du[np.isfinite(du)]) + eps
        S = np.exp(- (D / rbf_sigma) ** 2)
    elif method == "maxminus":
        m = np.nanmax(D)
        S = (m - D) / (m + eps)
    elif method == "inv":
        S = 1.0 / (1.0 + D)
    elif method == "neg":
        S = -D
    else:
        raise ValueError("Unknown method. Choose from {'rbf','maxminus','inv','neg'}.")
    np.fill_diagonal(S, 1.0)  # self-similarity
    return S


# -----------------------------------------------------------------------------
# Age-based developmental model matrices
# -----------------------------------------------------------------------------
@dataclass
class DevModels:
    nearest_neighbor: np.ndarray
    convergence: np.ndarray
    divergence: np.ndarray


def build_dev_models(
        age: np.ndarray,
        normalize: bool = True
) -> DevModels:
    """
    Build three developmental model matrices (higher = predicts greater similarity):
    - Nearest-neighbor: high when |Δage| is small (similar maturity)
      M_nn = max(age) - |age_i - age_j|
    - Convergence: high when both are older (group tuning with age)
      M_conv = min(age_i, age_j)
    - Divergence: high when both are younger (young are more similar)
      M_div = max(age) - (age_i + age_j)/2
    """
    a = np.asarray(age, float).reshape(-1)
    n = a.size
    A_i = np.repeat(a[:, None], n, axis=1)
    A_j = A_i.T
    amax = np.nanmax(a)

    M_nn = amax - np.abs(A_i - A_j)
    M_conv = np.minimum(A_i, A_j)
    M_div = amax - 0.5 * (A_i + A_j)

    if normalize:
        def _norm(M):
            mu = np.nanmean(triu_vectorize(M, 1))
            sd = np.nanstd(triu_vectorize(M, 1))
            return (M - mu) / (sd if sd > 0 else 1.0)

        M_nn = _norm(M_nn)
        M_conv = _norm(M_conv)
        M_div = _norm(M_div)

    # Set diagonal to NaN so they do not contribute in vectorization if desired
    np.fill_diagonal(M_nn, np.nan)
    np.fill_diagonal(M_conv, np.nan)
    np.fill_diagonal(M_div, np.nan)

    return DevModels(M_nn, M_conv, M_div)


# -----------------------------------------------------------------------------
# Partial Mantel (vector-level residualization)
# -----------------------------------------------------------------------------
def residualize_against(X: np.ndarray, nuisances: List[np.ndarray]) -> np.ndarray:
    """
    Regress a vector X on multiple nuisance vectors and return residuals.
    Each nuisance is z-scored; intercept included implicitly.
    """
    y = np.asarray(X, float).reshape(-1, 1)
    if not nuisances:
        return y.ravel()
    Z = np.column_stack([zscore(v) for v in nuisances])
    # Add intercept
    Z = np.column_stack([np.ones(Z.shape[0]), Z])
    # OLS residuals
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    yhat = Z @ beta
    resid = (y - yhat).ravel()
    return resid


# -----------------------------------------------------------------------------
@dataclass
class ISRSAresult:
    rho_nn: float
    rho_conv: float
    rho_div: float
    p_nn: float
    p_conv: float
    p_div: float
    fdr_reject: Dict[str, bool]
    fdr_crit: float
    boot_summary: pd.DataFrame  # rows: model, mean, ci_low, ci_high
    best_model: str
    details: Dict[str, np.ndarray]  # e.g., perm distributions, boot distributions


def _vectorize_valid_pairs(S: np.ndarray, M: np.ndarray, mask_user: Optional[np.ndarray] = None) -> Tuple[
    np.ndarray, np.ndarray]:
    """Get paired upper-tri vectors with finite mask (and optional user mask)."""
    s = triu_vectorize(S, 1)
    m = triu_vectorize(M, 1)
    valid = np.isfinite(s) & np.isfinite(m)
    if mask_user is not None:
        valid &= mask_user
    return s[valid], m[valid]


def _spearman_with_nuisance(s_vec: np.ndarray, m_vec: np.ndarray, nuis_vecs: Optional[List[np.ndarray]]) -> float:
    """Compute Spearman after residualizing both s_vec and m_vec on nuisances (partial Mantel style)."""
    if nuis_vecs:
        nuisances = [nv for nv in nuis_vecs]
        s_res = residualize_against(s_vec, nuisances)
        m_res = residualize_against(m_vec, nuisances)
        return _rank_spearman(s_res, m_res)
    else:
        return _rank_spearman(s_vec, m_vec)


def _permute_within_strata(age: np.ndarray, strata: Optional[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Permute globally or within strata blocks."""
    a = np.array(age, float).copy()
    if strata is None:
        return rng.permutation(a)
    g = np.asarray(strata)
    out = a.copy()
    for lab in np.unique(g):
        idx = np.where(g == lab)[0]
        out[idx] = out[idx][rng.permutation(len(idx))]
    return out


def is_rsa_fit(
        S: np.ndarray,
        age: np.ndarray,
        n_perm: int = 5000,
        perm_sided: str = "greater",  # 'two', 'greater', 'less'
        strata: Optional[np.ndarray] = None,  # e.g., site labels for within-site permutation
        nuis_mats: Optional[List[np.ndarray]] = None,  # list of NxN nuisance matrices for partial Mantel
        boot_B: int = 1000,
        boot_frac: float = 0.8,
        boot_seed: int = 42
) -> ISRSAresult:
    """
    Run IS-RSA on a similarity matrix S and age vector.
    Returns observed rhos, permutation p-values, FDR across models, and bootstrap summaries.
    """
    n = S.shape[0]
    assert S.shape == (n, n), "S must be square."
    age = np.asarray(age, float).reshape(-1)
    assert age.size == n, "age length must match S."

    # Build development models
    models = build_dev_models(age, normalize=True)

    # Vectorize S and each model (upper-tri)
    s_nn, m_nn = _vectorize_valid_pairs(S, models.nearest_neighbor)
    s_cv, m_cv = _vectorize_valid_pairs(S, models.convergence)
    s_dv, m_dv = _vectorize_valid_pairs(S, models.divergence)

    # Prepare nuisance vectors (vectorized upper-tri) if any
    nuis_vecs = None
    if nuis_mats:
        nuis_vecs = []
        for G in nuis_mats:
            g = triu_vectorize(G, 1)
            # Replace non-finite with median to avoid dropping pairs
            gf = np.where(np.isfinite(g), g, np.nan)
            med = np.nanmedian(gf)
            gf = np.where(np.isfinite(g), g, med)
            nuis_vecs.append(gf.astype(float))

    # Observed Spearman correlations (partial if nuisances provided)
    rho_nn = _spearman_with_nuisance(s_nn, m_nn, nuis_vecs)
    rho_conv = _spearman_with_nuisance(s_cv, m_cv, nuis_vecs)
    rho_div = _spearman_with_nuisance(s_dv, m_dv, nuis_vecs)

    # Permutation test: shuffle age (globally or within strata), rebuild model each time
    rng = np.random.default_rng(123)
    perm_nn = np.empty(n_perm, dtype=float)
    perm_cv = np.empty(n_perm, dtype=float)
    perm_dv = np.empty(n_perm, dtype=float)

    for b in range(n_perm):
        a_perm = _permute_within_strata(age, strata, rng)
        M_perm = build_dev_models(a_perm, normalize=True)
        # Vectorize
        _, mnn = _vectorize_valid_pairs(S, M_perm.nearest_neighbor)
        _, mcv = _vectorize_valid_pairs(S, M_perm.convergence)
        _, mdv = _vectorize_valid_pairs(S, M_perm.divergence)
        # Partial mantel (residualize) if needed
        perm_nn[b] = _spearman_with_nuisance(s_nn, mnn, nuis_vecs)
        perm_cv[b] = _spearman_with_nuisance(s_cv, mcv, nuis_vecs)
        perm_dv[b] = _spearman_with_nuisance(s_dv, mdv, nuis_vecs)

    def p_from_perm(obs: float, dist: np.ndarray, sided: str) -> float:
        if sided == "two":
            return (np.sum(np.abs(dist) >= np.abs(obs)) + 1.0) / (dist.size + 1.0)
        elif sided == "greater":
            return (np.sum(dist >= obs) + 1.0) / (dist.size + 1.0)
        elif sided == "less":
            return (np.sum(dist <= obs) + 1.0) / (dist.size + 1.0)
        else:
            raise ValueError("perm_sided must be 'two', 'greater', or 'less'.")

    p_nn = p_from_perm(rho_nn, perm_nn, perm_sided)
    p_conv = p_from_perm(rho_conv, perm_cv, perm_sided)
    p_div = p_from_perm(rho_div, perm_dv, perm_sided)

    # FDR across the three models
    p_all = np.array([p_nn, p_conv, p_div])
    rej, crit = bh_fdr(p_all, alpha=0.05)
    fdr_map = {"nearest_neighbor": bool(rej[0]),
               "convergence": bool(rej[1]),
               "divergence": bool(rej[2])}

    # Bootstrap (subsample without replacement): stability of rho for each model
    rngb = np.random.default_rng(boot_seed)
    m = max(4, int(np.floor(boot_frac * n)))
    boot_nn, boot_cv, boot_dv = [], [], []

    # Precompute index for faster slicing
    for t in range(boot_B):
        idx = np.sort(rngb.choice(n, size=m, replace=False))
        # Subset S and ages
        Sb = S[np.ix_(idx, idx)]
        ab = age[idx]
        Mb = build_dev_models(ab, normalize=True)

        s_nn_b, m_nn_b = _vectorize_valid_pairs(Sb, Mb.nearest_neighbor)
        s_cv_b, m_cv_b = _vectorize_valid_pairs(Sb, Mb.convergence)
        s_dv_b, m_dv_b = _vectorize_valid_pairs(Sb, Mb.divergence)

        # Nuisances need to be subset as well (if any)
        nuis_vecs_b = None
        if nuis_mats:
            nuis_vecs_b = [triu_vectorize(G[np.ix_(idx, idx)], 1) for G in nuis_mats]

        boot_nn.append(_spearman_with_nuisance(s_nn_b, m_nn_b, nuis_vecs_b))
        boot_cv.append(_spearman_with_nuisance(s_cv_b, m_cv_b, nuis_vecs_b))
        boot_dv.append(_spearman_with_nuisance(s_dv_b, m_dv_b, nuis_vecs_b))

    def _summ(v: List[float], q=(2.5, 97.5)) -> Tuple[float, float, float]:
        arr = np.asarray(v, float)
        return float(np.nanmean(arr)), float(np.nanpercentile(arr, q[0])), float(np.nanpercentile(arr, q[1]))

    mn_nn, lo_nn, hi_nn = _summ(boot_nn)
    mn_cv, lo_cv, hi_cv = _summ(boot_cv)
    mn_dv, lo_dv, hi_dv = _summ(boot_dv)

    boot_summary = pd.DataFrame({
        "model": ["nearest_neighbor", "convergence", "divergence"],
        "rho_obs": [rho_nn, rho_conv, rho_div],
        "p_perm": [p_nn, p_conv, p_div],
        "rho_boot_mean": [mn_nn, mn_cv, mn_dv],
        "rho_boot_ci_low": [lo_nn, lo_cv, lo_dv],
        "rho_boot_ci_high": [hi_nn, hi_cv, hi_dv],
        "fdr_sig": [fdr_map["nearest_neighbor"], fdr_map["convergence"], fdr_map["divergence"]]
    }).sort_values("rho_boot_mean", ascending=False, ignore_index=True)

    best_model = boot_summary.loc[0, "model"]

    details = {
        "perm_nn": perm_nn, "perm_conv": perm_cv, "perm_div": perm_dv,
        "boot_nn": np.asarray(boot_nn), "boot_conv": np.asarray(boot_cv), "boot_div": np.asarray(boot_dv)
    }

    return ISRSAresult(
        rho_nn=rho_nn, rho_conv=rho_conv, rho_div=rho_div,
        p_nn=p_nn, p_conv=p_conv, p_div=p_div,
        fdr_reject=fdr_map, fdr_crit=crit,
        boot_summary=boot_summary,
        best_model=best_model,
        details=details
    )


# -----------------------------------------------------------------------------
# Multiple Regression on (Dis)similarity Matrices (MRM)
# -----------------------------------------------------------------------------
@dataclass
class MixRegResult:
    method: str
    coefs: np.ndarray  # beta or weights (length = n_predictors)
    intercept: float  # 0 for simplex mode
    r2: float
    r2_perm: Optional[np.ndarray]  # permuted nulls for R^2
    p_global: Optional[float]  # permutation p for R^2
    p_coefs: Optional[np.ndarray]  # permutation p for |beta| (OLS only)
    beta_perm: Optional[np.ndarray]  # permuted nulls of betas
    partial_r2: Optional[np.ndarray]  # semi-partial R^2 by leave-one-out predictors
    yhat_mat: np.ndarray


def matrix_mixture_regression(
        M_obs: np.ndarray,
        M_nn: np.ndarray,
        M_con: np.ndarray,
        M_div: np.ndarray,
        *,
        method: str = "ols",  # 'ols' | 'ridge' | 'simplex'
        ridge_alpha: float = 1e-2,  # L2 penalty for 'ridge'
        standardize: bool = True,  # z-score y and each X column (recommended for OLS/Ridge)
        n_perm: int = 5000,
        strata: Optional[np.ndarray] = None,  # e.g., site labels for within-site permutations
        seed: int = 42
) -> MixRegResult:
    """
    Regress an observed similarity matrix on three developmental model matrices.
    method='ols'   -> unconstrained OLS with intercept
    method='ridge' -> ridge with intercept (alpha=ridge_alpha)
    method='simplex' -> nonnegative weights summing to 1 (no intercept), i.e., convex mixture
    """
    # 1) vectorize
    n = M_obs.shape[0]
    y = triu_vectorize(M_obs, k=1)
    X = np.column_stack([triu_vectorize(M_nn, k=1),
                         triu_vectorize(M_con, k=1),
                         triu_vectorize(M_div, k=1)])

    # handle NaNs consistently
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[valid];
    X = X[valid, :]

    # 2) standardize (recommended for OLS/Ridge; for simplex we use unit-norm to neutralize scale)
    if method in ("ols", "ridge"):
        if standardize:
            y_s = zscore(y)
            X_s = np.column_stack([zscore(X[:, k]) for k in range(X.shape[1])])
        else:
            y_s, X_s = y, X
        # design with intercept
        Xd = np.column_stack([np.ones(X_s.shape[0]), X_s])

        if method == "ols":
            beta, *_ = np.linalg.lstsq(Xd, y_s, rcond=None)
        else:
            # ridge with intercept (no penalty on intercept)
            I = np.eye(Xd.shape[1]);
            I[0, 0] = 0.0
            XtX = Xd.T @ Xd
            beta = np.linalg.solve(XtX + ridge_alpha * I, Xd.T @ y_s)

        b0, b = float(beta[0]), beta[1:].astype(float)
        yhat = Xd @ beta
        # R^2 on standardized scale equals Pearson^2 between y_s and yhat (since includes intercept)
        ss_res = np.sum((y_s - yhat) ** 2)
        ss_tot = np.sum((y_s - np.mean(y_s)) ** 2)
        r2_obs = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # partial (semi-partial) R^2 for each predictor
        partial = np.zeros(3, float)
        for k in range(3):
            keep = [j for j in range(3) if j != k]
            Xk = np.column_stack([np.ones(X_s.shape[0]), X_s[:, keep]])
            bk = np.linalg.lstsq(Xk, y_s, rcond=None)[0]
            yhat_k = Xk @ bk
            ss_res_k = np.sum((y_s - yhat_k) ** 2)
            r2_k = 1.0 - ss_res_k / ss_tot
            partial[k] = r2_obs - r2_k

        # permutation (QAP): permute node labels in M_obs only
        rng = np.random.default_rng(seed)
        r2_perm = np.full(n_perm, np.nan)
        beta_perm = np.full((n_perm, 3), np.nan)
        if n_perm > 0:
            for b_iter in tqdm(range(n_perm), desc='Permutation on significance of betas for developmental model'):
                p = permute_labels(n, strata, rng)
                M_perm = M_obs[np.ix_(p, p)]
                y_perm_all = triu_vectorize(M_perm, k=1)[valid]  # same valid mask applies
                y_perm = zscore(y_perm_all) if standardize else y_perm_all
                # fit on permuted y
                bp = np.linalg.lstsq(Xd, y_perm, rcond=None)[0]
                yhat_p = Xd @ bp
                ss_res_p = np.sum((y_perm - yhat_p) ** 2)
                ss_tot_p = np.sum((y_perm - np.mean(y_perm)) ** 2)
                r2_perm[b_iter] = 1.0 - ss_res_p / ss_tot_p if ss_tot_p > 0 else 0.0
                beta_perm[b_iter, :] = bp[1:]

        p_global = (np.sum(r2_perm >= r2_obs) + 1.0) / (n_perm + 1.0) if np.isfinite(r2_perm).all() else None
        p_coefs = np.array([(np.sum(np.abs(beta_perm[:, k]) >= abs(b[k])) + 1.0) / (n_perm + 1.0)
                            for k in range(3)], float) if np.isfinite(beta_perm).all() else None

        # pack
        yhat_vec = yhat
        yhat_mat = unvec_ut(yhat_vec if not standardize else
                            (yhat * np.nanstd(y) + np.nanmean(y)), n)

        return MixRegResult(
            method=method, coefs=b, intercept=b0, r2=r2_obs,
            r2_perm=r2_perm, p_global=p_global,
            p_coefs=p_coefs, beta_perm=beta_perm,
            partial_r2=partial, yhat_vec=yhat_vec, yhat_mat=yhat_mat
        )

    elif method == "simplex":
        # unit-norm columns so weights are comparable; no intercept, w>=0, sum w=1
        Xn = X / np.maximum(np.linalg.norm(X, axis=0, ord=2), 1e-12)

        def obj(w):
            r = y - Xn @ w
            return 0.5 * np.dot(r, r)

        # constraints: sum w = 1, w >= 0
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, None)] * Xn.shape[1]
        w0 = np.ones(Xn.shape[1]) / Xn.shape[1]
        res = optimize.minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
        w = res.x
        yhat = Xn @ w
        # pseudo R^2 (on original y scale)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_obs = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # permutation test for global R^2 and weight stability
        rng = np.random.default_rng(seed)
        r2_perm = np.empty(n_perm, float)
        w_perm = np.empty((n_perm, 3), float)
        for b_iter in range(n_perm):
            p = permute_labels(n, strata, rng)
            y_perm = triu_vectorize(M_obs[np.ix_(p, p)], k=1)[valid]

            def obj_p(wp):
                r = y_perm - Xn @ wp
                return 0.5 * np.dot(r, r)

            res_p = optimize.minimize(obj_p, w0, method='SLSQP', bounds=bounds, constraints=cons)
            wp = res_p.x
            yh = Xn @ wp
            ss_res_p = np.sum((y_perm - yh) ** 2)
            ss_tot_p = np.sum((y_perm - np.mean(y_perm)) ** 2)
            r2_perm[b_iter] = 1.0 - ss_res_p / ss_tot_p if ss_tot_p > 0 else 0.0
            w_perm[b_iter, :] = wp

        p_global = (np.sum(r2_perm >= r2_obs) + 1.0) / (n_perm + 1.0)
        # (optional) coefficient-level p (one-sided for contribution >=)
        p_coefs = np.array([(np.sum(w_perm[:, k] >= w[k]) + 1.0) / (n_perm + 1.0)
                            for k in range(3)], float)

        yhat_vec = yhat
        yhat_mat = unvec_ut(yhat_vec, n)

        return MixRegResult(
            method=method, coefs=w, intercept=0.0, r2=r2_obs,
            r2_perm=r2_perm, p_global=p_global, p_coefs=p_coefs,
            partial_r2=None, yhat_vec=yhat_vec, yhat_mat=yhat_mat
        )

    else:
        raise ValueError("method must be 'ols', 'ridge', or 'simplex'")


# -----------------------------------------------------------------------------
# Decomposition and diffusion embedding
# -----------------------------------------------------------------------------
def center_kernel(K):
    """Double-center a similarity matrix (kernel centering)."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def fro_inner(A, B):
    """Frobenius inner product <A,B> = sum(A*B)."""
    return float(np.sum(np.nan_to_num(A, nan=1) * np.nan_to_num(B, nan=1)))


def gram_schmidt_fro(templates, tol: float = 1e-10, reorth: bool = True):
    """
    Orthonormalize a list of same-shaped matrices under the Frobenius inner product.

    Parameters
    ----------
    templates : list[np.ndarray]
        List of matrices (same shape) T_1, T_2, ..., T_m.
    tol : float
        Tolerance for declaring a new direction degenerate.
    reorth : bool
        If True, perform a second orthogonalization pass (stabilizes numerics).

    Returns
    -------
    U : list[np.ndarray]
        Orthonormal basis (under Frobenius). Each U[i] has <U[i],U[j]>=delta_ij.
    R : np.ndarray, shape (r, m)
        Upper-triangular-like coefficient matrix such that
        T_j ≈ sum_{i=0..r-1} R[i, j] * U[i], where r = rank under Frobenius-orthogonality.

    Notes
    -----
    - If some templates are (near) linearly dependent in Frobenius sense,
      they will NOT spawn a new basis vector; their column in R will be
      expressed in the existing basis (shorter column padded with zeros).
    - This is a Frobenius-QR: columns of R give coordinates of T_j in the U-basis.
    """
    U = []  # orthonormal basis (list of matrices)
    R_cols = []  # store columns of R with dynamic length
    eps = 1e-12

    for T in templates:
        W = T.astype(float, copy=True)
        coeffs = []

        # first pass
        for Uj in U:
            c = fro_inner(W, Uj)
            coeffs.append(c)
            W -= c * Uj

        # optional second pass (re-orthogonalization)
        if reorth and len(U) > 0:
            for i, Uj in enumerate(U):
                c2 = fro_inner(W, Uj)
                coeffs[i] += c2
                W -= c2 * Uj

        nrm2 = fro_inner(W, W)
        if nrm2 <= tol:
            # Degenerate: no new basis vector; just record coefficients in existing basis.
            R_cols.append(coeffs)  # will be zero-padded later
            continue

        nrm = np.sqrt(max(nrm2, eps))
        U_new = W / nrm
        U.append(U_new)
        # Column for this template includes the new component as the last coefficient.
        R_cols.append(coeffs + [nrm])

    # Assemble R with zero-padding to final rank (rows = len(U), cols = m)
    r = len(U)
    m = len(templates)
    R = np.zeros((r, m), dtype=float)
    for j, col in enumerate(R_cols):
        R[:len(col), j] = col

    return U, R


def nearest_psd(K, eps=1e-10):
    """Eigenvalue clipping to the nearest PSD (simple repair)."""
    K = 0.5 * (K + K.T)
    w, V = eigh(K)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T


def to_affinity(K, mode="auto", rbf_sigma=None):
    """
    Ensure a non-negative affinity for diffusion/spectral embedding.
    If K already looks like similarity in [0,1], just scale and clip.
    """
    A = 0.5 * (K + K.T)
    if mode == "rbf":
        # convert a "distance-like" K to affinity via RBF
        D = A
        du = D[np.triu_indices_from(D, 1)]
        if rbf_sigma is None:
            rbf_sigma = np.nanmedian(du[np.isfinite(du)]) + 1e-9
        A = np.exp(-(D / rbf_sigma) ** 2)
    else:
        # min-max to [0,1]
        m, M = np.nanmin(A), np.nanmax(A)
        if M > m:
            A = (A - m) / (M - m)
        A[A < 0] = 0.0
        np.fill_diagonal(A, 1.0)
    return nearest_psd(A)


def diffusion_embed(A, n_components=3, alpha=0.5):
    """
    Basic diffusion map (Coifman & Lafon): build P = D^(-alpha) A D^(-alpha), row-normalize, eigendecompose.
    Returns eigenvalues (sorted desc, skip trivial 1), and embedding (N x n_components).
    """
    A = np.maximum(A, 0)
    A = 0.5 * (A + A.T)
    d = np.sum(A, axis=1, keepdims=True) + 1e-12
    D_alpha = d ** (-alpha)
    K = D_alpha * A * D_alpha.T
    row_sums = np.sum(K, axis=1, keepdims=True) + 1e-12
    P = K / row_sums

    # eigen-decomposition
    w, V = eigh(0.5 * (P + P.T))  # symmetrized for numerical stability
    idx = np.argsort(w)[::-1]
    w = w[idx];
    V = V[:, idx]
    # drop the first trivial eigenvector (eigenvalue ~1)
    vals = w[1:n_components + 1]
    vecs = V[:, 1:n_components + 1]
    # diffusion coordinates (t=1 here): psi_i = lambda_i * phi_i
    emb = vecs * vals
    return vals, emb


# -----------------------------------------------------------------------------
# projection / decomposition
# -----------------------------------------------------------------------------
def matched_component_linear(M_obs, templates, method="ols", simplex=False):
    """
    Project M_obs onto span{templates} in edge-space.
    - method='ols'/'ridge' (ridge with small L2 via Tikhonov by hand if needed);
    - or simplex=True to do NNLS then normalize weights to sum=1.
    Returns beta (weights), M_match, M_res, per-template matched parts.
    """
    n = M_obs.shape[0]
    y = triu_vectorize(M_obs)
    X = np.column_stack([triu_vectorize(T) for T in templates])

    if simplex:
        # NNLS weights (nonnegative); normalize to sum=1 if possible
        w = nnls(X, y)[0]
        s = w.sum()
        beta = w / s if s > 1e-12 else w
    else:
        # plain OLS in edge space
        beta, *_ = lstsq(X, y, rcond=None)

    M_match = sum(b * T for b, T in zip(beta, templates))
    M_res = M_obs - M_match
    parts = [float(b) * T for b, T in zip(beta, templates)]
    return beta, M_match, M_res, parts


def matched_component_orthoproject(M_obs, templates):
    """
    Frobenius-orthogonal projection: Gram-Schmidt the templates, then project M_obs.
    This yields mutually orthogonal matched parts.
    """
    U, _ = gram_schmidt_fro(templates)  # orthonormal basis under Fro inner product
    coeffs = [fro_inner(M_obs, Ui) for Ui in U]  # coefficients along each Ui
    parts_ortho = [c * Ui for c, Ui in zip(coeffs, U)]
    M_match = sum(parts_ortho)
    M_res = M_obs - M_match
    return coeffs, parts_ortho, M_match, M_res


def frob_cos(A, B, center=True):
    """Frobenius cosine similarity with optional double-centering."""
    A_ = center_kernel(A) if center else A
    B_ = center_kernel(B) if center else B
    num = np.sum(A_ * B_)
    den = np.sqrt(np.sum(A_ ** 2) * np.sum(B_ ** 2)) + 1e-12
    return float(num / den) if den > 0 else np.nan


def qap_perm_score(Kcomp, Tmpl, n_perm=1000, seed=0, strata=None, center=True):
    """
    QAP: permute subject labels (optionally within strata), recompute Frobenius cosine.
    Return (obs, p_two_tailed).
    """
    rng = np.random.default_rng(seed)
    n = Kcomp.shape[0]
    obs = frob_cos(Kcomp, Tmpl, center=center)
    cnt = 0
    for _ in tqdm(range(n_perm), desc='Permutation on correspondence btw embedding and dev_model'):
        if strata is None:
            perm = rng.permutation(n)
        else:
            # within-strata shuffle
            perm = np.arange(n)
            for lab in np.unique(strata):
                idx = np.where(strata == lab)[0]
                perm[idx] = rng.permutation(idx)
        Kp = Kcomp[np.ix_(perm, perm)]
        sc = frob_cos(Kp, Tmpl, center=center)
        if abs(sc) >= abs(obs) - 1e-12:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return obs, p


# -----------------------------------------------------------------------------
# main: confirm which template each diffusion component matches
# -----------------------------------------------------------------------------
def confirm_dm_components(Y, lambdas, templates, template_names=("nearest", "convergence", "divergence"),
                          weight_by_lambda=True, center=True, n_perm=2000, strata=None):
    """
    Y: (N x K) diffusion coordinates (t=0版本的特征向量即可)
    lambdas: (K,) corresponding eigenvalues (用于可选加权)
    templates: list of template matrices [M_nn, M_con, M_div] (N x N each)
    Returns:
        scores: (K x T) Frobenius cosines (after sign alignment to be positive if wanted)
        pvals: (K x T) QAP two-tailed p-values
        signs: (K,) applied sign flips on Y[:,k] to make the best-matching template positive
    """
    N, K = Y.shape
    T = len(templates)
    scores = np.zeros((K, T))
    pvals = np.ones((K, T))
    signs = np.ones(K)

    # 1) build component kernels
    comps = []
    for k in range(K):
        v = Y[:, k].copy()
        Kk = np.outer(v, v)
        if weight_by_lambda:
            Kk = (lambdas[k] if np.isfinite(lambdas[k]) else 1.0) * Kk
        comps.append(Kk)

    # 2) for each component, choose sign by the best-aligned template (so that score >= 0)
    for k, Kk in enumerate(comps):
        raw = [frob_cos(Kk, Tmpl, center=center) for Tmpl in templates]
        j_best = int(np.nanargmax(np.abs(raw)))
        sgn = np.sign(raw[j_best]) if raw[j_best] != 0 else 1.0
        signs[k] = sgn
        comps[k] = sgn * Kk  # flip rank-1 kernel equivalently to flipping v

    # 3) compute cosines + QAP p for each template
    for k, Kk in enumerate(comps):
        for t, Tmpl in enumerate(templates):
            sc, pv = qap_perm_score(Kk, Tmpl, n_perm=n_perm, seed=1234 + k, strata=strata, center=center)
            scores[k, t] = sc
            pvals[k, t] = pv

    return {"scores": scores, "pvals": pvals, "signs": signs, "template_names": template_names}


def component_distance_alignment(Y, templates, from_similarity=True):
    """
    For each component k, correlate edgewise distances D_k(i,j) = |v_k(i)-v_k(j)|
    with model "distances" (1 - normalized similarity if from_similarity=True).
    Returns Spearman r (K x T).
    """
    from scipy.stats import spearmanr
    N, K = Y.shape
    iu, ju = np.triu_indices(N, 1)
    R = np.zeros((K, len(templates)))

    # model "distance"
    def sim_to_dist(M):
        m, M = np.nanmin(M), np.nanmax(M)
        S01 = (M - m) / (M - m + 1e-12)
        return 1.0 - S01

    D_models = [sim_to_dist(M) if from_similarity else M for M in templates]
    Dm_ut = [Dm[iu, ju] for Dm in D_models]

    for k in range(K):
        v = Y[:, k]
        Dk = np.abs(v[iu] - v[ju])
        for t in range(len(templates)):
            r, _ = spearmanr(Dk, Dm_ut[t])
            R[k, t] = r
    return R


# -----------------------------------------------------------------------------
# MRM based on sliding window along age grid
# -----------------------------------------------------------------------------
def _make_age_grid(age: np.ndarray, n_grid: int = 15, lo_q: float = 5.0, hi_q: float = 95.0) -> np.ndarray:
    """Percentile-based age grid to avoid sparse tails."""
    a = np.asarray(age).ravel()
    lo = np.percentile(a, lo_q)
    hi = np.percentile(a, hi_q)
    return np.linspace(lo, hi, n_grid)


def _subset_mats(M: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Induce an NxN submatrix with consistent subject order."""
    return M[np.ix_(idx, idx)]


@dataclass
class LocalAgeMixResult:
    centers: np.ndarray  # (G,) age centers used
    half_width: float  # window half-width (years)
    method: str  # 'ols' | 'ridge' | 'simplex'
    betas: np.ndarray  # (G, 3) beta_nn, beta_con, beta_div (same order as call)
    intercepts: np.ndarray  # (G,) intercept (0 for simplex)
    r2: np.ndarray  # (G,) local R^2
    partial_r2: np.ndarray  # (G,)
    p_global: np.ndarray  # (G,) permutation p for global R^2
    p_coefs: np.ndarray  # (G,) permutation p for coefficients (simplex: one-sided as in your impl)
    n_in_win: np.ndarray  # (G,) sample size per window
    idx_list: List[np.ndarray]  # subject indices per window (for reproducibility / further analysis)


def local_age_varying_mix(
    M_obs: np.ndarray,
    M_nn: np.ndarray,
    M_con: np.ndarray,
    M_div: np.ndarray,
    age: np.ndarray,
    *,
    grid: Optional[np.ndarray] = None,  # if None, use percentile grid
    h: float = 2.0,                     # half-width of age window (in years)
    min_n: int = 30,                    # minimum subjects in a window to run a fit
    method: str = "ridge",              # 'ols' | 'ridge' | 'simplex'
    ridge_alpha: float = 1e-2,
    standardize: bool = True,
    n_perm: int = 2000,                 # typically smaller than global run for speed
    strata: Optional[np.ndarray] = None,# site labels for within-site permutations
    seed: int = 42
) -> LocalAgeMixResult:
    """
    Slide an age window along `grid`, at each center subset subjects with |age - a0| <= h,
    and fit the mixture regression by RE-USING `matrix_mixture_regression` on the induced submatrices.
    """
    age = np.asarray(age).ravel()
    N = len(age)
    assert M_obs.shape == (N, N), "Matrix and age length must match."

    if grid is None:
        grid = _make_age_grid(age, n_grid=15, lo_q=5.0, hi_q=95.0)

    G = len(grid)
    betas = np.full((G, 3), np.nan, float)
    intercepts = np.full(G, np.nan, float)
    r2 = np.full(G, np.nan, float)
    partial_r2 = np.full((G, 3), np.nan, float)
    p_global = np.full(G, np.nan, float)
    p_coefs = np.full((G, 3), np.nan, float)
    n_in_win = np.zeros(G, int)
    idx_list: List[np.ndarray] = []

    for g, a0 in enumerate(grid):
        print(f"\nMRM on age window: [{a0-h}, {a0+h}]] --> ")
        idx = np.where(np.abs(age - a0) <= h)[0]
        idx_list.append(idx)
        n_in_win[g] = len(idx)
        if len(idx) < min_n:
            continue  # leave as NaN for this grid point

        sub = np.ix_(idx, idx)
        sub_strata = None if strata is None else np.asarray(strata)[idx]

        # RE-USE your regression function on the submatrices
        res = matrix_mixture_regression(
            M_obs[sub], M_nn[sub], M_con[sub], M_div[sub],
            method=method,
            ridge_alpha=ridge_alpha,
            standardize=standardize,
            n_perm=n_perm,
            strata=sub_strata,
            seed=seed
        )

        # store results
        # NOTE: `coefs` order in your function was [X[:,0], X[:,1], X[:,2]] corresponding to (M_nn, M_con, M_div)
        betas[g, :] = res.coefs
        partial_r2[g, :] = res.partial_r2
        intercepts[g] = res.intercept
        r2[g] = res.r2
        p_global[g] = res.p_global if res.p_global is not None else np.nan
        if res.p_coefs is not None:
            p_coefs[g, :] = res.p_coefs

    return LocalAgeMixResult(
        centers=np.asarray(grid),
        half_width=h,
        method=method,
        betas=betas,
        intercepts=intercepts,
        r2=r2,
        partial_r2=partial_r2,
        p_global=p_global,
        p_coefs=p_coefs,
        n_in_win=n_in_win,
        idx_list=idx_list
    )

@dataclass
class BetaCIs:
    grid: np.ndarray
    beta_mean: np.ndarray    # (G, 3)
    beta_lo: np.ndarray      # (G, 3)
    beta_hi: np.ndarray      # (G, 3)
    r2_mean: np.ndarray

def _fit_edge_regression(y, X, method="ridge", ridge_alpha=1e-2, standardize=True, w: Optional[np.ndarray]=None):
    """
    Weighted OLS/Ridge with intercept on edge-level design.
    y: (m,), X: (m,3), optional weights w: (m,)
    Standardizes y and each X column (weighted if w is provided).
    Returns (beta[3], R2).
    """
    y = np.asarray(y, float); X = np.asarray(X, float)
    eps = 1e-12

    if w is None:
        if standardize:
            y = (y - y.mean()) / (y.std() + eps)
            X = (X - X.mean(0)) / (X.std(0) + eps)
        Xd = np.column_stack([np.ones(len(y)), X])
        Yw, Xw = y, Xd
        mu = y.mean()
        w_vec = None
    else:
        w = np.asarray(w, float)
        sw = np.sqrt(w)
        # weighted standardization
        def wmean(a): return np.sum(w * a) / (np.sum(w) + eps)
        def wstd(a):  mu = wmean(a); return np.sqrt(np.sum(w * (a - mu)**2) / (np.sum(w) + eps))
        if standardize:
            y = (y - wmean(y)) / (wstd(y) + eps)
            for k in range(X.shape[1]):
                X[:, k] = (X[:, k] - wmean(X[:, k])) / (wstd(X[:, k]) + eps)
        Xd = np.column_stack([np.ones(len(y)), X])
        Xw = Xd * sw[:, None]
        Yw = y * sw
        mu = wmean(y)
        w_vec = w

    # solve
    if method == "ridge":
        I = np.eye(Xw.shape[1]); I[0, 0] = 0.0  # no penalty on intercept
        beta = np.linalg.solve(Xw.T @ Xw + ridge_alpha * I, Xw.T @ Yw)
    else:  # 'ols'
        beta, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
    b = beta[1:]

    # R^2 (weighted if w provided)
    yhat = Xd @ beta
    if w_vec is None:
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - mu)**2)
    else:
        ss_res = np.sum(w_vec * (y - yhat)**2)
        ss_tot = np.sum(w_vec * (y - mu)**2)
    r2 = 1.0 - ss_res / (ss_tot + eps)
    return b, r2

def bootstrap_beta_curves(
    M_obs, M_nn, M_con, M_div, age,
    grid=None, h=2.0, min_n=30, B=500,
    method="ridge", ridge_alpha=1e-2, standardize=True,
    q: Tuple[float, float]=(2.5, 97.5), seed=7,
    scheme: str = "edge"  # 'edge' or 'node'
) -> BetaCIs:
    """
    Sliding-window bootstrap for beta(a) curves.
    scheme='edge': resample upper-triangular edges with replacement.
    scheme='node': draw node weights from Dirichlet and use WLS with edge weights w_i w_j.
    """
    rng = np.random.default_rng(seed)
    age = np.asarray(age).ravel()
    if grid is None:
        lo, hi = np.percentile(age, 5), np.percentile(age, 95)
        grid = np.linspace(lo, hi, 15)

    G = len(grid)
    beta_arr = np.full((G, B, 3), np.nan)
    r2_arr   = np.full((G, B), np.nan)
    beta_mean = np.full((G, 3), np.nan)
    r2_mean = np.full(G, np.nan)

    for g, a0 in enumerate(grid):
        idx = np.where(np.abs(age - a0) <= h)[0]
        n = len(idx)
        if n < min_n:
            continue

        # submatrices for this window
        sub = np.ix_(idx, idx)
        M0 = np.asarray(M_obs[sub], float)
        M1 = np.asarray(M_nn[sub], float)
        M2 = np.asarray(M_con[sub], float)
        M3 = np.asarray(M_div[sub], float)

        # build edge-level arrays (upper triangle only)
        iu, ju = np.triu_indices(n, 1)
        y_full = M0[iu, ju]
        X_full = np.column_stack([M1[iu, ju], M2[iu, ju], M3[iu, ju]])

        # one non-boot estimate for mean curves
        b0, r0 = _fit_edge_regression(y_full, X_full, method=method, ridge_alpha=ridge_alpha, standardize=standardize)
        beta_mean[g, :] = b0
        r2_mean[g] = r0

        m = len(y_full)
        for b in tqdm(range(B), desc=f'Bootstrapping for CIs of betas of dev_model on age window: [{a0-h}, {a0+h}]]'):
            if scheme == "edge":
                # sample edges with replacement (no self-pairs ever)
                samp = rng.integers(0, m, size=m)
                yb = y_full[samp]
                Xb = X_full[samp, :]
                bb, r2b = _fit_edge_regression(yb, Xb, method=method, ridge_alpha=ridge_alpha, standardize=standardize)
            elif scheme == "node":
                # Bayesian bootstrap-of-weights on nodes -> edge weights w_i w_j
                w_node = rng.dirichlet(np.ones(n))
                w_edge = w_node[iu] * w_node[ju]
                bb, r2b = _fit_edge_regression(y_full, X_full, method=method, ridge_alpha=ridge_alpha,
                                               standardize=standardize, w=w_edge)
            else:
                raise ValueError("scheme must be 'edge' or 'node'")

            beta_arr[g, b, :] = bb
            r2_arr[g, b] = r2b

    # percentile CIs over bootstrap replicates (ignore NaNs)
    lo = np.nanpercentile(beta_arr, q[0], axis=1)
    hi = np.nanpercentile(beta_arr, q[1], axis=1)
    return BetaCIs(grid=np.asarray(grid),
                   beta_mean=beta_mean,
                   beta_lo=lo, beta_hi=hi,
                   r2_mean=r2_mean)

def build_match_and_residual(M_obs, M_nn, M_con, M_div, beta_vec):
    """Construct M_match and M_res given weights beta=[b_nn,b_con,b_div]."""
    M_match = beta_vec[0]*M_nn + beta_vec[1]*M_con + beta_vec[2]*M_div
    M_res   = M_obs - M_match
    # symmetrize defensively
    M_match = 0.5*(M_match + M_match.T)
    M_res   = 0.5*(M_res + M_res.T)
    np.fill_diagonal(M_match, 0.0); np.fill_diagonal(M_res, 0.0)
    return M_match, M_res

def local_distinctiveness(D, age, h=2.0, kernel="gaussian"):
    """
    Age-aligned distinctiveness: per-subject weighted average dissimilarity to others.
    If you start from similarity, convert to dissimilarity before calling.
    """
    age = np.asarray(age).ravel()
    n = len(age); D = np.asarray(D, float)
    out = np.full(n, np.nan)
    for i in range(n):
        diff = np.abs(age - age[i])
        if kernel == "gaussian":
            w = np.exp(-(diff**2)/(2*h**2))
        else:
            w = (diff <= h).astype(float)
        w[i] = 0.0
        s = w.sum()
        if s > 0:
            w /= s
            out[i] = np.sum(w * D[i,:])
    return out  # length N

def curves_for_match_vs_res(
    M_obs, M_nn, M_con, M_div, age,
    beta_curve: np.ndarray,  # (G,3) beta at each grid
    grid: np.ndarray,
    h_local=2.0,
    from_similarity=True
) -> Dict[str, np.ndarray]:
    """
    For each age center, build M_match using beta(center), define dissimilarity matrices,
    compute age-aligned distinctiveness for M_match and M_res, then average by small bins around the center.
    """
    age = np.asarray(age).ravel()
    N = len(age); G = len(grid)
    # store the age-binned mean distinctiveness at each center
    d_match = np.full(G, np.nan)
    d_res   = np.full(G, np.nan)

    # helper: similarity->dissimilarity
    def to_D(M):
        m, Mx = np.nanmin(M), np.nanmax(M)
        if Mx > m:
            S01 = (M - m) / (Mx - m)
        else:
            S01 = M - m
        return 1.0 - S01  # larger = more dissimilar

    for g, a0 in enumerate(grid):
        b = beta_curve[g, :]
        M_match, M_res = build_match_and_residual(M_obs, M_nn, M_con, M_div, b)

        Dm = to_D(M_match) if from_similarity else M_match
        Dr = to_D(M_res)   if from_similarity else M_res

        # per-subject local distinctiveness (age-aligned)
        dm_i = local_distinctiveness(Dm, age, h=h_local, kernel="gaussian")
        dr_i = local_distinctiveness(Dr, age, h=h_local, kernel="gaussian")

        # average around the current center (narrow bin ±0.5y)
        idx_bin = np.where(np.abs(age - a0) <= 0.5)[0]
        if len(idx_bin) > 3:
            d_match[g] = np.nanmean(dm_i[idx_bin])
            d_res[g]   = np.nanmean(dr_i[idx_bin])
    return {"grid": grid, "d_match": d_match, "d_res": d_res}

# ========================================
# Optional helpers: make nuisance pairwise matrices quickly
# ========================================
def nuisance_from_labels(labels: np.ndarray) -> np.ndarray:
    """
    Build an NxN *dissimilarity* matrix from categorical labels.
    0 if same label, 1 if different (can be used as nuisance in partial Mantel).
    """
    lab = np.asarray(labels)
    n = lab.size
    L_i = np.repeat(lab[:, None], n, axis=1)
    L_j = L_i.T
    G = (L_i != L_j).astype(float)
    np.fill_diagonal(G, np.nan)
    return G

def nuisance_from_continuous(x: np.ndarray, mode: str = "absdiff") -> np.ndarray:
    """
    Build an NxN *dissimilarity* matrix from a continuous covariate.
    - 'absdiff' : |xi - xj|
    - 'sqdiff'  : (xi - xj)^2
    - 'mean'    : 0.5*(xi + xj)  (e.g., to control for cohort mean-level)
    """
    v = np.asarray(x, float).reshape(-1)
    n = v.size
    X_i = np.repeat(v[:, None], n, axis=1)
    X_j = X_i.T
    if mode == "absdiff":
        G = np.abs(X_i - X_j)
    elif mode == "sqdiff":
        G = (X_i - X_j) ** 2
    elif mode == "mean":
        G = 0.5 * (X_i + X_j)
    else:
        raise ValueError("mode must be 'absdiff', 'sqdiff', or 'mean'.")
    np.fill_diagonal(G, np.nan)
    return G

def compare_models_by_boot(out, model_A="convergence", model_B="nearest_neighbor", ci=(2.5,97.5)):
    """
    Paired bootstrap comparison: is rho_A > rho_B?
    Returns mean diff and percentile CI; if CI>0 => A > B.
    """
    key = {"nearest_neighbor": "boot_nn",
           "convergence": "boot_conv",
           "divergence": "boot_div"}
    A = out.details[key[model_A]]
    B = out.details[key[model_B]]
    n = min(len(A), len(B))
    diff = (A[:n] - B[:n])
    mean_diff = float(np.mean(diff))
    lo, hi = np.percentile(diff, ci[0]), np.percentile(diff, ci[1])
    print(f"rho ({model_A} - {model_B}) = {mean_diff:.4f} "
          f"[{ci[0]}%, {ci[1]}% CI: {lo:.4f}, {hi:.4f}]")
    return mean_diff, lo, hi

# ========================================
# visualization helpers
# ========================================
def plot_perm_and_boot(
    out,
    model="convergence",
    *,
    do_bootstrap=False,
    # histogram + vline customization
    perm_hist_kwargs=None,
    boot_hist_kwargs=None,
    perm_vline_kwargs=None,
    boot_vline_kwargs=None,
    # auto-truncation controls (permutation panel only)
    truncate_perm_if_far=True,
    perm_core_quantiles=(0.5, 99.5),
    z_far_threshold=None,
    perm_core_pad_frac=0.05,
    obs_window_std=0.20,
    break_gap_frac=0.02,
    right_width_ratio=1.2,
    figsize_perm=(6.5, 3.6),
    figsize_boot=(5, 3.5),
    bins=40,
    density=True,
    title_prefix=True,
    # slash style (drawn in display/pixel coordinates)
    slash_length_px=14,   # each slash length (pixels)
    slash_sep_px=8,       # vertical separation between two slashes (pixels)
    slash_angle_deg=60,   # tilt angle (degrees)
    slash_color="k",
    slash_lw=1.3,
):
    """Permutation-null (with robust broken x-axis) + bootstrap histogram.
    Slashes are placed exactly on the x-axis break using display (pixel) coords.
    """
    # -------- map fields --------
    m2key = {"nearest_neighbor": ("perm_nn","boot_nn","rho_nn"),
             "convergence"      : ("perm_conv","boot_conv","rho_conv"),
             "divergence"       : ("perm_div","boot_div","rho_div")}
    perm_key, boot_key, rho_key = m2key[model]
    perm = np.asarray(out.details[perm_key], float)
    boot = np.asarray(out.details[boot_key], float)
    rho_obs = float(getattr(out, rho_key))

    # -------- defaults --------
    if perm_hist_kwargs is None:
        perm_hist_kwargs = dict(alpha=0.9, edgecolor="none")
    if boot_hist_kwargs is None:
        boot_hist_kwargs = dict(alpha=0.9, edgecolor="none")
    if perm_vline_kwargs is None:
        perm_vline_kwargs = dict(color="r", ls="--", lw=2)
    if boot_vline_kwargs is None:
        boot_vline_kwargs = dict(color="k", ls="--", lw=2)

    hist_counts, hist_edges = np.histogram(perm, bins=bins, density=density)

    def _despine(ax, left=True, bottom=True, right=True, top=True):
        if left:  ax.spines["left"].set_visible(False)
        if bottom: ax.spines["bottom"].set_visible(False)
        if right: ax.spines["right"].set_visible(False)
        if top: ax.spines["top"].set_visible(False)

    # helper: draw two *parallel* slashes anchored to the x-axis in pixel coords
    def _draw_slashes_bottom(fig, ax, side="right"):
        """
        Draw EXACTLY two short parallel slashes // at the x-axis of `ax`.
        side='right' puts them at the right inner edge; side='left' at the left inner edge.
        Positions are computed in display/pixel coordinates, then mapped to figure coords.
        """
        # anchor at axes inner corner on the x-axis in *display* coords
        # (1,0) = right-bottom, (0,0) = left-bottom in axes coordinates
        anchor_axes = (1.0, 0.0) if side == "right" else (0.0, 0.0)
        x0_px, y0_px = ax.transAxes.transform(anchor_axes)  # -> display (pixels)

        # build half-length vector along the desired angle (in pixels)
        ang = np.deg2rad(slash_angle_deg)
        hx, hy = 0.5 * slash_length_px * np.cos(ang), 0.5 * slash_length_px * np.sin(ang)
        # two *parallel* slashes: same angle, same center x, shifted only in y by sep
        for k in range(1):
            y_shift = k * slash_sep_px
            A_disp = (x0_px - hx, y0_px - hy + y_shift)
            B_disp = (x0_px + hx, y0_px + hy + y_shift)
            # convert display -> figure coords for a Line2D with fig.transFigure
            A_fig = fig.transFigure.inverted().transform(A_disp)
            B_fig = fig.transFigure.inverted().transform(B_disp)
            line = Line2D([A_fig[0], B_fig[0]], [A_fig[1], B_fig[1]],
                          transform=fig.transFigure,
                          lw=slash_lw, clip_on=False, color=slash_color)
            fig.add_artist(line)

        # -------- permutation panel --------

    mu = np.nanmean(perm)
    sd = np.nanstd(perm) if np.nanstd(perm) > 0 else 1e-9
    qlo, qhi = np.nanpercentile(perm, perm_core_quantiles)
    core_w = max(qhi - qlo, 1e-12)
    pad = perm_core_pad_frac * core_w

    far_by_quantile = (rho_obs < qlo) or (rho_obs > qhi)
    far_by_z = (abs((rho_obs - mu) / sd) >= (z_far_threshold or np.inf))
    do_break = truncate_perm_if_far and (
        far_by_quantile and far_by_z if z_far_threshold is not None else far_by_quantile)

    if do_break:
        left_min = qlo - pad
        left_max = qhi + pad
        half_w = max(obs_window_std * sd, 1e-12)
        right_min = rho_obs - half_w
        right_max = rho_obs + half_w

        # ensure a visible gap between the two x-ranges
        full_min = min(np.min(perm), right_min)
        full_max = max(np.max(perm), right_max)
        full_w = max(full_max - full_min, 1e-9)
        gap = break_gap_frac * full_w
        if left_max >= right_min - gap:
            mid = 0.5 * (left_max - gap + right_min)
            left_max = mid - 0.5 * gap
            right_min = mid + 0.5 * gap

        fig, (axL, axR) = plt.subplots(
            1, 2, sharey=True, figsize=figsize_perm,
            gridspec_kw={"width_ratios": [2.0, right_width_ratio]}
        )

        # Left panel (keep left/bottom spines)
        axL.hist(perm, bins=hist_edges, density=density, **perm_hist_kwargs)
        axL.set_xlim(left_min, left_max)
        axL.set_xlabel("rho");
        axL.set_ylabel("density")
        axL.xaxis.set_major_locator(MaxNLocator(nbins=5))

        if title_prefix:
            axL.set_title(f"Permutation null ({model})\nobs rho = {rho_obs:.3f}")
        _despine(axL, left=False, bottom=False, right=True, top=True)

        # Right panel: keep only bottom spine (x-axis line), hide others
        axR.hist(perm, bins=hist_edges, density=density, **perm_hist_kwargs)
        axR.set_xlim(right_min, right_max)
        axR.set_xlabel("rho")
        axR.axvline(rho_obs, **perm_vline_kwargs)
        _despine(axR, left=True, bottom=False, right=True, top=True)
        axR.tick_params(labelleft=False, labelright=False, left=False)

        # finalize layout FIRST, then draw slashes exactly at the break
        plt.tight_layout()
        fig.canvas.draw()  # update all transforms

        # draw one pair on the right edge of the left panel
        _draw_slashes_bottom(fig, axL, side="right")
        # and one pair on the left edge of the right panel
        _draw_slashes_bottom(fig, axR, side="left")

    else:
        fig = plt.figure(figsize=figsize_perm)
        ax = plt.gca()
        ax.hist(perm, bins=hist_edges, density=density, **perm_hist_kwargs)
        xmin = min(qlo - pad, rho_obs)
        xmax = max(qhi + pad, rho_obs)
        xr = xmax - xmin
        ax.set_xlim(xmin - 0.05 * xr, xmax + 0.05 * xr)
        ax.axvline(rho_obs, **perm_vline_kwargs)
        if title_prefix:
            ax.set_title(f"Permutation null ({model})\nobs rho = {rho_obs:.3f}")
        ax.set_xlabel("rho");
        ax.set_ylabel("density")
        _despine(ax, left=False, bottom=False, right=True, top=True)
        plt.tight_layout()

    # -------- bootstrap panel --------
    if do_bootstrap:
        axb = plt.figure(figsize=figsize_boot)
        axb = plt.gca()
        axb.hist(boot, bins=bins, density=density, **boot_hist_kwargs)
        axb.axvline(np.nanmean(boot), **boot_vline_kwargs)
        if title_prefix:
            axb.set_title(f"Bootstrapped rhos ({model})\nmean = {np.nanmean(boot):.3f}")
        axb.set_xlabel("rho");
        axb.set_ylabel("density")
        _despine(axb, left=False, bottom=False, right=True, top=True)
        plt.tight_layout()


def age_sort_index(age, ascending=True):
    """Return permutation that sorts age."""
    idx = np.argsort(np.asarray(age).ravel())
    return idx if ascending else idx[::-1]


def reorder_square(M, idx):
    """Apply the same permutation to rows and cols of a square matrix."""
    return M[np.ix_(idx, idx)]


def plot_dev_models_and_similarity(S, age, reorder_by_age=True, figsize=(10, 9)):
    """
    Plot (1) three developmental model matrices and (2) the observed similarity matrix.
    If reorder_by_age=True, matrices are jointly reordered by ascending age.
    Returns (fig, axes, idx) where idx is the permutation used.
    """
    age = np.asarray(age).ravel()
    n = len(age)
    assert S.shape == (n, n), "S and age must match."

    # Optionally reorder by age for visualization
    if reorder_by_age:
        idx = age_sort_index(age, ascending=True)
        age_plot = age[idx]
        S_plot = reorder_square(S, idx)
    else:
        idx = np.arange(n)
        age_plot = age
        S_plot = S

    # Build models on the (possibly) reordered age
    models = build_dev_models(age_plot, normalize=True)

    # Prepare 2x2 grid: NN / Conv / Div / Observed S
    mats = [
        ("Nearest-neighbor (z)", models.nearest_neighbor),
        ("Convergence (z)", models.convergence),
        ("Divergence (z)", models.divergence),
        ("Observed similarity", S_plot)
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    for ax, (title, M) in zip(axes, mats):
        im = ax.imshow(M, origin="upper", aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("Subjects (sorted by age)")
        ax.set_ylabel("Subjects (sorted by age)")
        # Sparse age tick labels
        ticks = np.linspace(0, n - 1, num=min(6, n), dtype=int)
        ax.set_xticks(ticks);
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{age_plot[t]:.1f}" for t in ticks], rotation=45, ha="right")
        ax.set_yticklabels([f"{age_plot[t]:.1f}" for t in ticks])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, axes, idx


##% PREPARE DATA INPUT
# -----------------------------------------------------------------------------
fpath_df = '/public/home/dtngrui/fmri_analysis/data/beh/er_profiles.csv'
df = pd.read_csv(fpath_df, sep='\t')

feature_info = {
    'efforts_ER': {'type': 'numeric'},
    'success_ER': {'type': 'numeric'},
    'ERD-p': {'type': 'numeric'},
    'ERD-n': {'type': 'numeric'},
    'ERQ-ES': {'type': 'numeric'},
    'ERQ-CR': {'type': 'numeric'}
}

##% CALCULATE INTER-SUBJECT SIMILARITY MATRIX
# -----------------------------------------------------------------------------
# 1) Distance matrices
rng = np.random.default_rng(0)
Ds = compute_all_distance_matrices(
    df, feature_info=feature_info, robust_maha=True, shrinkage='lw', ridge_alpha=0.1
)
print("Mahalanobis shape:", Ds['mahalanobis'].shape)
print("Spearman shape:", Ds['spearman'].shape)
print("Gower shape:", Ds['gower'].shape)

# 2) Bootstrap stability for distinctiveness & average dissimilarity (Mahalanobis as example)
res_boot = bootstrap_individual_metrics(
    df, metric='mahalanobis',
    B=1000, subsample_frac=0.8,
    random_state=123,
    feature_info=feature_info,
    robust=True, shrinkage='lw', ridge_alpha=0.1
)
print(res_boot.head())

##% MODELING AGE EFFECT ON INTER-SUBJECT DIFFERENCE OF ER PROFILES
# -----------------------------------------------------------------------------
# 1) Convert distance -> similarity (any monotonic transform is fine for rank-based Mantel)
S = distance_to_similarity(Ds['mahalanobis'], method="rbf")  # or 'maxminus'/'inv'/'neg'

# 2) Run IS-RSA with within-site permutation + partial Mantel
res = is_rsa_fit(
    S=S,
    age=np.array(df.age),
    n_perm=1000,  # increase for final analysis
    perm_sided="greater",  # commonly 'greater' if expecting positive rho
    strata=None,  # permute age within sites (None if no stratification)
    nuis_mats=[],  # [] if you don't want partial Mantel
    boot_B=1000,  # bootstrap iterations for stability
    boot_frac=0.8,  # subsample fraction
    boot_seed=123
)

##% COMPARE MODELS
# -----------------------------------------------------------------------------
# which developmental model has better fits with observed inter-subject similarity
mean_m1_m2, lo_m1_m2, hi_m1_m2 = compare_models_by_boot(res, "convergence", "nearest_neighbor")

##% MULTIPLE REGRESSION ON MATRIX (RSA-BASED REGRESSION)
# -----------------------------------------------------------------------------
# inter-subject similarity of ER profiles
Ds = compute_all_distance_matrices(
    df,
    feature_info=feature_info,
    robust_maha=True,
    shrinkage='lw',
    ridge_alpha=0.1,
)
S = distance_to_similarity(Ds['mahalanobis'], method="rbf")  # or 'maxminus'/'inv'/'neg'

# reorder matrix according to age
idx = age_sort_index(np.array(df.age), ascending=True)
S_sorted = reorder_square(S, idx)
age_sorted = np.asarray(df.age).ravel()[idx]

# generate three developmental model matrices
M_dev = build_dev_models(df['age'], normalize=True)
M_dev_sorted = build_dev_models(age_sorted, normalize=True)

# 1) MRM on whole age range (6-18)
whole = matrix_mixture_regression(
    M_obs=S,
    M_nn=M_dev.nearest_neighbor, M_con=M_dev.convergence, M_div=M_dev.divergence,
    method="ridge",
    n_perm=1000,
)

# 2) MRM via sliding window along age grid
#    15 age centers with sliding window of half width of 2.5
local = local_age_varying_mix(
    M_obs=S,
    M_nn=M_dev.nearest_neighbor, M_con=M_dev.convergence, M_div=M_dev.divergence,
    age=df['age'],
    grid=None,  # auto percentile-based grid
    h=2.5,  # 2-year half window
    min_n=40,  # make sure each window has enough subjects
    method="ridge",  # 'ols' | 'ridge' | 'simplex'
    ridge_alpha=1e-2,
    standardize=True,
    n_perm=1000,
    strata=None,  # or site id
    seed=1
)

# 3) bootstrap for reliable CI of betas on developmental models
betaCI = bootstrap_beta_curves(
    M_obs=S,
    M_nn=M_dev.nearest_neighbor, M_con=M_dev.convergence, M_div=M_dev.divergence,
    age=df['age'],
    grid=None, h=2.5,
    B=1000,
    method="ridge", ridge_alpha=1e-2, standardize=True,
    q=(2.5, 97.5),
    seed=7,
    scheme="node"  # 'edge' or 'node'
)

# 4) age-related curves of local distinctiveness for inter-subject difference
curve_localDist = curves_for_match_vs_res(
    M_obs=S,
    M_nn=M_dev.nearest_neighbor, M_con=M_dev.convergence, M_div=M_dev.divergence,
    age=df['age'],
    h_local=2.5,
    beta_curve=betaCI.beta_mean,
    grid=betaCI.grid,
    from_similarity=True,
)

##% TRY DECOMPOSITION AND DIFFUSION EMBEDDING OF INTER-SUBJECT SIMILARITY MATRIX
# -----------------------------------------------------------------------------
# decomposition
templates = [M_dev.nearest_neighbor, M_dev.convergence]
coeffs, parts_ortho, M_match, M_res = matched_component_orthoproject(S, templates)


def r_fro(A, B):
    return fro_inner(A, B) / (np.sqrt(fro_inner(A, A)) * np.sqrt(fro_inner(B, B)) + 1e-12)


print("Fro-alignment with M_con:", r_fro(M_match, M_dev.convergence))
print("Fro-alignment with M_nn:", r_fro(M_match, M_dev.nearest_neighbor))

# diffusion mapping to extract first 3 components
A_match = to_affinity(nearest_psd(np.nan_to_num(M_match, nan=1)))
A_res = to_affinity(nearest_psd(np.nan_to_num(M_res, nan=1)))

vals_m, emb_m = diffusion_embed(A_match, n_components=3)  # (N x 3) coords
vals_r, emb_r = diffusion_embed(A_res, n_components=3)

# which developmental model dominates diffusion component
out = confirm_dm_components(
    emb_m, vals_m,
    [np.nan_to_num(M_dev.nearest_neighbor, nan=1),
     np.nan_to_num(M_dev.convergence, nan=1),
     np.nan_to_num(M_dev.divergence, nan=1)],
    template_names=("nearest", "convergence", "divergence"),
    weight_by_lambda=True, center=True, n_perm=1000, strata=None
)
print("Frobenius cosines:\n", out["scores"])
print("QAP p-values:\n", out["pvals"])

##% VISUALIZATIONS OF MAIN RESULTS
# -----------------------------------------------------------------------------
# 1) Reorder matrix by age for visualization
idx = age_sort_index(np.array(df.age), ascending=True)
S_sorted = reorder_square(S, idx)
age_sorted = np.asarray(df.age).ravel()[idx]

# 2) Plot model matrices + observed similarity (sorted)
plot_dev_models_and_similarity(S_sorted, age_sorted, reorder_by_age=False)

# 3) permutation and bootstrap visualization
plot_perm_and_boot(
    res, model="convergence",
    do_bootstrap=False, figsize_perm=(5, 5), figsize_boot=(5, 5),
    perm_hist_kwargs={"alpha": 0.75, "edgecolor": None},
    boot_hist_kwargs={"alpha": 0.75, "edgecolor": None},
    perm_vline_kwargs={"color": "r", "ls": "--", "lw": 1.5},
    boot_vline_kwargs={"color": "r", "ls": "--", "lw": 1.5},
    truncate_perm_if_far=True,
    perm_core_quantiles=(0.5, 99.5),
    z_far_threshold=3.0,
    obs_window_std=0.25,
    break_gap_frac=0.02,
    right_width_ratio=1.2,
)

# 4) scatter btw inter-subject diff and age
df_temp = pd.DataFrame({
    'age': np.array(df.age),
    'distinctiveness': res_boot['distinctiveness_mean'],
})

g = sns.jointplot(x="age", y="distinctiveness", data=df_temp,
                  kind="reg",
                  truncate=False,
                  # xlim=(0, 60), ylim=(0, 12),
                  joint_kws=dict(
                      scatter_kws=dict(color="r", alpha=0.75, s=80),  # points style
                      line_kws=dict(color="gray", linewidth=3, alpha=0.9)  # regression line style
                      # you can also set ci=, order=, robust=, etc., here if needed
                  ),
                  color="k", height=5)
g.fig.set_size_inches(8, 6)  # width=10", height=6"
g.fig.tight_layout()  # or: g.fig.set_constrained_layout(True)
plt.show()


##% TEST CELL (TEMP FUNCTIONS)
# -----------------------------------------------------------------------------
def bootstrap_beta_curves(
        M_obs, M_nn, M_con, M_div, age,
        B=500,
        method="ridge", ridge_alpha=1e-2, standardize=True,
        scheme: str = "edge",  # 'edge' or 'node'
        q: Tuple[float, float] = (2.5, 97.5),
        seed=7,
) -> BetaCIs:
    """
    Sliding-window bootstrap for beta(a) curves.
    scheme='edge': resample upper-triangular edges with replacement.
    scheme='node': draw node weights from Dirichlet and use WLS with edge weights w_i w_j.
    """
    rng = np.random.default_rng(seed)
    age = np.asarray(age).ravel()

    beta_arr = np.full((B, 3), np.nan)
    r2_arr = np.full((B), np.nan)
    betas_obs = np.full((1, 3), np.nan)
    # r2_mean  = np.full(G, np.nan)

    # build edge-level arrays (upper triangle only)
    n = M_obs.shape[0]
    iu, ju = np.triu_indices(n, 1)
    y_full = triu_vectorize(M_obs, k=1)
    X_full = np.column_stack([triu_vectorize(M_nn, k=1),
                              triu_vectorize(M_con, k=1),
                              triu_vectorize(M_div, k=1)])

    b0, r0 = _fit_edge_regression(y_full, X_full, method=method, ridge_alpha=ridge_alpha, standardize=standardize)
    betas_obs[0, :] = b0
    r2 = r0

    m = len(y_full)
    for b in tqdm(range(B), desc='Bootstrapping for CIs of betas of dev_model'):
        if scheme == "edge":
            # sample edges with replacement (no self-pairs ever)
            samp = rng.integers(0, m, size=m)
            yb = y_full[samp]
            Xb = X_full[samp, :]
            bb, r2b = _fit_edge_regression(yb, Xb, method=method, ridge_alpha=ridge_alpha, standardize=standardize)
        elif scheme == "node":
            # Bayesian bootstrap-of-weights on nodes -> edge weights w_ij = w_i * w_j
            w_node = rng.dirichlet(np.ones(n))
            w_edge = w_node[iu] * w_node[ju]
            bb, r2b = _fit_edge_regression(y_full, X_full, method=method, ridge_alpha=ridge_alpha,
                                           standardize=standardize, w=w_edge)
        else:
            raise ValueError("scheme must be 'edge' or 'node'")

        beta_arr[b, :] = bb
        r2_arr[b] = r2b

    # percentile CIs over bootstrap replicates (ignore NaNs)
    lo = np.nanpercentile(beta_arr, q[0], axis=0)
    hi = np.nanpercentile(beta_arr, q[1], axis=0)
    return BetaCIs(grid=np.asarray(age),
                   beta_mean=betas_obs,
                   beta_lo=lo, beta_hi=hi,
                   r2_mean=r2, beta_arr=beta_arr)


def plot_match_vs_res(curves, ax=None):
    x = curves["grid"]
    plt.figure(figsize=(7, 5))
    plt.plot(x, curves["d_match"], label="distinctiveness(M_match)")
    plt.plot(x, curves["d_res"], label="distinctiveness(M_residual)")
    plt.xlabel("Age")
    plt.ylabel("Age-aligned distinctiveness")
    plt.legend()
    plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()


def plot_r2(local_res, ax=None):
    x = local_res.centers;
    R2 = local_res.r2
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, R2, color='k')
        ax.set_xlabel("Age")
        ax.set_ylabel("Local R²")
        bsc.nice_axes(ax, which_grid=None)
        plt.tight_layout()

def plot_beta_with_ci(
        grid, beta_mean, beta_lo=None, beta_hi=None,
        labels=("nnt", "cov", "div"),
        colors=("#280038", "#229B55", "#C6C100"),
        ylabel=r"$\beta$ with bootstrap CI", xlabel="Age",
        ax=None,
):
    """
    Draw beta(a) curves with optional bootstrap percentile CIs.
    p_coefs: array (G,3) of permutation p-values for coefficient-wise significance (optional).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    for k in range(beta_mean.shape[1]):
        ax.plot(grid, beta_mean[:, k], lw=2.2, label=labels[k], color=colors[k])
        if (beta_lo is not None) and (beta_hi is not None):
            ax.fill_between(grid,
                            beta_lo[:, k], beta_hi[:, k],
                            alpha=0.25, color=colors[k], linewidth=0)

    ax.axhline(y=0,
                xmin=0.05, xmax=0.95,
                color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel)
    bsc.nice_axes(ax, which_grid=None)
    ax.legend(ncol=1, frameon=False)

def compute_degree(S, drop_diag=True, make_symmetric=True, nan_to_zero=True,
                normalize=None):
    """
        Fast node degree (row-sum) from a subject-by-subject similarity matrix S.

        Parameters
        ----------
        S : (N,N) array-like
            Subject-by-subject similarity (can be any real-valued affinity).
        drop_diag : bool
            If True, set diagonal to 0 before summing.
        make_symmetric : bool
            If True, symmetrize by (S + S.T)/2 to stabilize numerics.
        nan_to_zero : bool
            If True, NaN/inf are replaced by 0 before summing.
        normalize : {'mean', 'zscore', None}
            - None: raw degree (sum over row).
            - 'mean': average similarity per subject (= sum/(N-1)).
            - 'zscore': z-scored across subjects.

        Returns
        -------
        deg : (N,) np.ndarray
            Degree per subject.
    """
    S = np.asarray(S, dtype=float)
    if make_symmetric:
        S = 0.5 * (S + S.T)
    if drop_diag:
        np.fill_diagonal(S, 0.0)
    if nan_to_zero:
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    deg = S.sum(axis=1)

    if normalize == 'mean':
        n = S.shape[0]
        deg = deg / max(1, n - 1)
    elif normalize == 'zscore':
        mu, sd = np.mean(deg), np.std(deg) + 1e-12
        deg = (deg - mu) / sd

    return deg

def degree_age_corr(S, age, method='pearson', normalize='mean',
                    drop_diag=True, make_symmetric=True, nan_to_zero=True):
    """
        Convenience: compute degree vector and its correlation with age.
        Returns
        -------
        deg : (N,) np.ndarray
        r, p : float
            Correlation (Pearson/Spearman) and its parametric p-value.
    """
    age = np.asarray(age).ravel()
    deg = compute_degree(S, drop_diag=drop_diag, make_symmetric=make_symmetric,
                            nan_to_zero=nan_to_zero, normalize=normalize)
    m = np.isfinite(age) & np.isfinite(deg)
    if method == 'spearman':
        r, p = stats.spearmanr(age[m], deg[m])
    else:
        r, p = stats.pearsonr(age[m], deg[m])
    return deg, r, p

def perm_pvalue_corr(x, y, n_perm=5000, strata=None, seed=0, method='pearson'):
    """
        Stratified label-permutation test for corr(x, y).
        Permutes y within strata (e.g., site) to control confounding.

        Returns
        -------
        obs_r : float
        p_perm : float  # two-sided permutation p-value
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    def _corr(a, b):
        if method == 'spearman':
            return stats.spearmanr(a, b)[0]
        return stats.pearsonr(a, b)[0]

    obs = _corr(x, y)
    n = len(y)
    if strata is None:
        strata = np.zeros(n, dtype=int)
    else:
        strata = np.asarray(strata)[mask]

    cnt = 0
    for _ in range(n_perm):
        yp = y.copy()
        for lab in np.unique(strata):
            idx = np.where(strata == lab)[0]
            yp[idx] = yp[idx][np.random.permutation(idx)]
        sc = _corr(x, yp)
        if abs(sc) >= abs(obs) - 1e-12:
            cnt += 1
    pval = (cnt + 1) / (n_perm + 1)
    return obs, pval

def edge_reg_mean_vs_diff(S, age, standardize=True):
    """
        Regress vec(S) on mean-age and age-difference (upper-tri edges).
        y ~ b0 + b1*mean_age + b2*(-|age_i - age_j|)  # sign so that larger = more-similar
        Returns standardized coefficients and partial R^2 for each predictor.
    """
    n = len(age);
    age = np.array(age)
    iu, ju = np.triu_indices(n, 1)
    y = S[iu, ju]

    mean_age = 0.5 * (age[iu] + age[ju])
    age_diff = -np.abs(age[iu] - age[ju])  # negative distance => higher for nearer ages

    X = np.column_stack([mean_age, age_diff])
    if standardize:
        y = zscore(y)
        X = np.column_stack([zscore(X[:, 0]), zscore(X[:, 1])])

    # add intercept
    Xd = np.column_stack([np.ones_like(y), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    ybar = np.mean(y)

    ss_tot = np.sum((y - ybar) ** 2)
    ss_res = np.sum((y - yhat) ** 2)
    r2_full = 1 - ss_res / (ss_tot + 1e-12)

    # partial R^2 via drop-one
    # drop mean-age
    Xd2 = np.column_stack([np.ones_like(y), X[:, 1]])
    b2, *_ = np.linalg.lstsq(Xd2, y, rcond=None)
    r2_drop_mean = 1 - np.sum((y - Xd2 @ b2) ** 2) / (ss_tot + 1e-12)
    # drop age-diff
    Xd1 = np.column_stack([np.ones_like(y), X[:, 0]])
    b1, *_ = np.linalg.lstsq(Xd1, y, rcond=None)
    r2_drop_diff = 1 - np.sum((y - Xd1 @ b1) ** 2) / (ss_tot + 1e-12)

    partial_r2_mean = r2_full - r2_drop_mean
    partial_r2_diff = r2_full - r2_drop_diff

    return {
        "beta_intercept": beta[0],
        "beta_mean_age": beta[1],
        "beta_age_diff": beta[2],
        "R2_full": r2_full,
        "partial_R2_mean_age": partial_r2_mean,
        "partial_R2_age_diff": partial_r2_diff
    }

##% PLOT FOR RESULTS OF MRM ON DYNAMIC AGE RANGE
# -----------------------------------------------------------------------------
bsc.set_pub_style(base_fontsize=15, font_family="DejaVu Sans")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

plot_beta_with_ci(
    local.centers, betaCI.beta_mean,
    beta_lo=betaCI.beta_lo, beta_hi=betaCI.beta_hi,
    labels=("nnb", "cov", "div"),
    colors=("#280038", "#229B55", "#C6C100"),
    ylabel=r"$\beta$ with bootstrap CI", xlabel="Age",
    ax=ax[0],
)

plot_r2(local, ax=ax[1])

bsc.permutation_boxplot(
    whole.beta_perm[:, 0], whole.coefs[0],
    box_width=0.07,
    gap=0.1,  # control distance between boxplot and density
    kde_height=0.1,  # peak height of density above its baseline
    fill=True,  # fill area under the density curve
    fill_alpha=0.3,
    xlabel="permuted nulls of beta",
    title=None
)

##%
betaCI_whole, betaCI_boots = bootstrap_beta_curves(
    M_obs=S,
    M_nn=M_dev.nearest_neighbor, M_con=M_dev.convergence, M_div=M_dev.divergence,
    age=df['age'],
    B=1000,
    method="ridge", ridge_alpha=1e-2, standardize=True,
    scheme="node",  # 'edge' or 'node'
    q=(2.5, 97.5),
    seed=7,
)

##% PLOT FOR RESULTS OF MRM ON WHOLE AGE RANGE
# -----------------------------------------------------------------------------
import seaborn as sns

beta_names = np.array(["nnb", "cov", "div"])
beta_value = betaCI_whole.beta_mean.copy()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

b = sns.barplot(x=beta_names,
                y=beta_value[0],
                palette=("#280038", "#229B55", "#C6C100"),
                alpha=0.75, ax=ax[0])
ax[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.8)
ax[0].set_xlabel("developmental models")
ax[0].set_ylabel(r"$\beta$ coefficients")
b.spines['top'].set_visible(False)
b.spines['right'].set_visible(False)

h = sns.histplot(whole.r2_perm, edgecolor=None, log_scale=True, ax=ax[1])
ax[1].axvline(x=whole.r2, color='r', linestyle='--', linewidth=1, alpha=0.8)
ax[1].set_xlabel("permuted $R^2$")
ax[1].set_ylabel("density")
h.spines['top'].set_visible(False)
h.spines['right'].set_visible(False)