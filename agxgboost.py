"""
AG-XGBoost: location-adaptive bandwidth geographically weighted XGBoost.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

try:
    import xgboost
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required: pip install -r requirements.txt") from exc

try:
    from sklearn.model_selection import GridSearchCV
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required: pip install -r requirements.txt") from exc

try:  # SHAP is optional
    import shap
except ImportError:  # pragma: no cover
    shap = None


# ----------------------------------------------------------------------------
# 0. Constants (values used in the paper)
# ----------------------------------------------------------------------------
SEED = 42
MIN_LOCAL_N = 5          # smallest admissible neighbourhood
BW_MIN = 5               # lower bound of the bandwidth interval
BW_STEP = 10             # coarse step of the stepwise (G-XGBoost) search
N_ANCHORS = 10           # number of anchors of the coarse scan
TOL = 3                  # golden-section convergence tolerance (in samples)
LOCAL_CV = 3             # CV folds for local tuning
GLOBAL_CV = 5            # CV folds for global tuning
M_FRACTION = 0.10        # representative locations: m = max(5, 10% of N)

# Default XGBoost configuration used as proxy model and fallback
BASE_PARAMS: Dict[str, object] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# Grid used to tune the hyper-parameters of the representative locations
LOCAL_GRID: Dict[str, List[object]] = {
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

# Grid used for the global (G-XGBoost baseline) configuration
GLOBAL_GRID: Dict[str, List[object]] = {
    "n_estimators": [30, 50, 100],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

# Small grid for the --demo / --fast mode
SMALL_GRID: Dict[str, List[object]] = {
    "n_estimators": [30],
    "learning_rate": [0.1],
    "max_depth": [3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

# Light-weight proxy configuration used by --fast (tutorial / smoke test only)
FAST_PARAMS: Dict[str, object] = {
    "n_estimators": 30,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


# ----------------------------------------------------------------------------
# 1. Spatial context: distance matrix, neighbour order, kernel weights
# ----------------------------------------------------------------------------
@dataclass
class SpatialContext:
    """Pairwise distances and neighbour ordering of the training locations.

    Attributes
    ----------
    dist : (N, N) array of Euclidean distances.
    order : (N, N) array; column i lists the locations sorted by distance to
        location i, with order[0, i] == i (the focal location itself).
    """

    dist: np.ndarray
    order: np.ndarray
    n: int

    def neighbours(self, i: int, bw: int) -> np.ndarray:
        """Indices of the ``bw`` nearest neighbours of i, excluding i itself."""
        bw = int(bw)
        if bw < 1:
            return np.empty(0, dtype=int)
        return self.order[1:bw + 1, i]

    def weights(self, i: int, bw: int) -> np.ndarray:
        """Adaptive bisquare kernel weights of the neighbours of i."""
        idx = self.neighbours(i, bw)
        d = self.dist[idx, i]
        h = float(d.max()) if d.size and d.max() > 0 else 1e-5
        return (1.0 - (d ** 2) / (h ** 2)) ** 2


def build_context(coords: np.ndarray) -> SpatialContext:
    """Build the distance matrix and the neighbour ordering.

    ``np.argsort`` may place a zero-distance duplicate first; the focal
    location is therefore moved explicitly to row 0 of every column.
    """
    dmat = distance_matrix(coords, coords)
    n = dmat.shape[0]
    order = np.argsort(dmat, axis=0, kind="stable")
    rows = np.arange(n)
    pos = np.argmax(order == rows, axis=0)
    cols = np.arange(n)
    first = order[0, :].copy()
    order[0, :] = order[pos, cols]
    order[pos, cols] = first
    if not np.array_equal(order[0, :], rows):
        raise ValueError("duplicate or NaN coordinates prevent a valid ordering")
    return SpatialContext(dist=dmat, order=order, n=n)


# ----------------------------------------------------------------------------
# 2. Local weighted XGBoost fitter
# ----------------------------------------------------------------------------
def make_estimator(params: Dict[str, object]) -> XGBRegressor:
    return XGBRegressor(**params, verbosity=0)


def fit_estimator(model: XGBRegressor, X: np.ndarray, y: np.ndarray,
                  sample_weight: Optional[np.ndarray] = None) -> XGBRegressor:
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


class LocalFitter:
    """Fit the weighted local model of (location, bandwidth) and return the
    leave-one-out prediction of the focal observation.

    Results are cached so that every (location, bandwidth) pair is fitted at
    most once; ``n_fits`` therefore equals the number of distinct evaluations.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, ctx: SpatialContext,
                 params: Dict[str, object], seed: int = SEED,
                 keep_models: bool = False):
        self.X, self.y, self.ctx = X, y, ctx
        self.params = dict(params)
        self.params.setdefault("random_state", seed)
        self.keep_models = keep_models
        self._cache: Dict[Tuple[int, int], Tuple[float, Optional[XGBRegressor], float]] = {}
        self.n_fits = 0

    def _fit_once(self, i: int, bw: int):
        idx = self.ctx.neighbours(i, bw)
        if len(idx) < MIN_LOCAL_N:
            return float("inf"), None, float("nan")
        if i in idx:  # leakage guard
            raise RuntimeError("focal observation in its own training set")
        w = self.ctx.weights(i, bw)
        model = fit_estimator(make_estimator(self.params), self.X[idx], self.y[idx], w)
        self.n_fits += 1
        pred = float(model.predict(self.X[[i]])[0])
        err = float((self.y[i] - pred) ** 2)
        return err, model, pred

    def _get(self, i: int, bw: int):
        key = (int(i), int(bw))
        if key not in self._cache:
            err, model, pred = self._fit_once(i, bw)
            self._cache[key] = (err, model if self.keep_models else None, pred)
        return self._cache[key]

    def error(self, i: int, bw: int) -> float:
        """Leave-one-out squared error CV_i(b)."""
        return self._get(i, bw)[0]

    def predict(self, i: int, bw: int) -> Tuple[Optional[XGBRegressor], float]:
        """(local model, leave-one-out prediction) at the selected bandwidth."""
        err, model, pred = self._get(i, bw)
        return model, pred


# ----------------------------------------------------------------------------
# 3. Bandwidth search
# ----------------------------------------------------------------------------
@dataclass
class SearchResult:
    bw: int
    n_evals: int
    trace: List[Tuple[int, float]] = field(default_factory=list)


def _cached_evaluator(evaluate):
    """Memoise the criterion so that no bandwidth is fitted twice."""
    cache: Dict[int, float] = {}

    def ev(b: int) -> float:
        b = int(b)
        if b not in cache:
            cache[b] = float(evaluate(b))
        return cache[b]

    return ev, cache


def step_search(evaluate, lo: int, hi: int, step: int = BW_STEP) -> SearchResult:
    """Stepwise search (G-XGBoost baseline): coarse grid, then local refinement."""
    ev, cache = _cached_evaluator(evaluate)
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        return SearchResult(lo, 1, [(lo, ev(lo))])
    coarse = list(range(lo, hi + 1, step))
    if coarse[-1] != hi:
        coarse.append(hi)
    best = min(coarse, key=ev)
    fine = range(max(lo, best - step), min(hi, best + step) + 1)
    best = min(fine, key=ev)
    return SearchResult(int(best), len(cache), sorted(cache.items()))


def golden_section_search(evaluate, lo: int, hi: int, tol: int = TOL,
                          cache: Optional[Dict[int, float]] = None) -> SearchResult:
    """Golden-section search on the integer bandwidth interval.

    ``cache`` may be shared with a previous stage (the coarse scan), so that
    bandwidths already evaluated are not fitted again.
    """
    if cache is None:
        cache = {}

    def ev(b: int) -> float:
        b = int(b)
        if b not in cache:
            cache[b] = float(evaluate(b))
        return cache[b]

    lo, hi = int(lo), int(hi)
    if hi <= lo:
        ev(lo)
        return SearchResult(lo, len(cache), sorted(cache.items()))

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    k1 = int(hi - phi * (hi - lo))
    k2 = int(lo + phi * (hi - lo))
    guard = 0
    while (hi - lo) > tol and guard < 200:
        guard += 1
        if ev(k1) < ev(k2):
            hi, k2 = k2, k1
            k1 = int(hi - phi * (hi - lo))
        else:
            lo, k1 = k1, k2
            k2 = int(lo + phi * (hi - lo))

    # Selection is restricted to the converged interval: the single-point LOO
    # criterion is noisy, and taking the global minimum over all evaluated
    # bandwidths selects degenerate neighbourhoods (b = 5) too often.
    best = min(range(lo, hi + 1), key=ev)
    return SearchResult(int(best), len(cache), sorted(cache.items()))


def coarse_golden_search(evaluate, lo: int, hi: int, n_anchors: int = N_ANCHORS,
                         tol: int = TOL) -> SearchResult:
    """Two-stage search: coarse anchor scan, then golden section.

    The anchors are equally spaced over [lo, hi] (both ends included).  The
    best anchor a* defines the search interval bounded by its left and right
    neighbours; the golden-section search is then run inside that interval,
    reusing the criterion values already computed for the anchors.
    """
    ev, cache = _cached_evaluator(evaluate)
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        return SearchResult(lo, 1, [(lo, ev(lo))])
    anchors = np.unique(np.round(np.linspace(lo, hi, max(3, n_anchors))).astype(int))
    errors = [ev(int(a)) for a in anchors]
    best_pos = int(np.argmin(errors))
    left = int(anchors[max(0, best_pos - 1)])
    right = int(anchors[min(len(anchors) - 1, best_pos + 1)])
    if right <= left:
        left, right = lo, hi
    return golden_section_search(evaluate, left, right, tol=tol, cache=cache)


# ----------------------------------------------------------------------------
# 4. Hyper-parameter configuration
# ----------------------------------------------------------------------------
def farthest_point_anchors(coords: np.ndarray, k: int) -> np.ndarray:
    """Farthest point sampling (k-centre greedy), deterministic.

    The first representative location is the observation closest to the
    centroid of the study area; each subsequent one is the observation whose
    distance to the already selected set is largest.
    """
    pts = np.asarray(coords, dtype=float)
    n = len(pts)
    k = max(1, min(int(k), n))
    if k >= n:
        return np.arange(n)
    centroid = pts.mean(axis=0)
    start = int(np.argmin(((pts - centroid) ** 2).sum(axis=1)))
    dmat = distance_matrix(pts, pts)
    anchors = [start]
    min_d = dmat[start].copy()
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        if nxt in anchors:
            nxt = int(np.argmax(np.where(np.isin(np.arange(n), anchors), -np.inf, min_d)))
        anchors.append(nxt)
        min_d = np.minimum(min_d, dmat[nxt])
    return np.array(sorted(anchors), dtype=int)


def effective_cv(n_local: int, cv: int = LOCAL_CV) -> int:
    """Use fewer folds when the neighbourhood is small (at least 2 points/fold)."""
    return int(max(2, min(cv, max(2, n_local) // 2)))


def tune_local(X: np.ndarray, y: np.ndarray, ctx: SpatialContext, i: int, bw: int,
               grid: Dict[str, List[object]], base: Dict[str, object],
               seed: int = SEED, cv: int = LOCAL_CV) -> Dict[str, object]:
    """Grid-search the hyper-parameters inside the weighted neighbourhood of i."""
    idx = ctx.neighbours(i, bw)
    if i in idx:
        raise RuntimeError("focal observation in its own tuning set")
    params = dict(base)
    params.setdefault("random_state", seed)
    gs = GridSearchCV(make_estimator(params), grid,
                      cv=effective_cv(len(idx), cv),
                      scoring="neg_mean_squared_error", n_jobs=1)
    gs.fit(X[idx], y[idx], sample_weight=ctx.weights(i, bw))
    return dict(gs.best_params_)


def tune_global(X: np.ndarray, y: np.ndarray, grid: Dict[str, List[object]],
                base: Dict[str, object], seed: int = SEED,
                cv: int = GLOBAL_CV) -> Dict[str, object]:
    """Grid-search a single configuration on the whole training set."""
    params = dict(base)
    params.setdefault("random_state", seed)
    gs = GridSearchCV(make_estimator(params), grid, cv=cv, scoring="r2", n_jobs=-1)
    gs.fit(X, y)
    return dict(gs.best_params_)


# ----------------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """R2, MAE and RMSE of the leave-one-out predictions."""
    resid = y_true - y_pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "n": int(len(y_true)),
    }


# ----------------------------------------------------------------------------
# 6. Main pipeline
# ----------------------------------------------------------------------------
def run(X: np.ndarray, y: np.ndarray, coords: np.ndarray,
        method: str = "agxgb", search: str = "coarse_golden",
        borrow: bool = True, n_anchors: int = N_ANCHORS, m_frac: float = M_FRACTION,
        grid: Optional[Dict[str, List[object]]] = None,
        base: Optional[Dict[str, object]] = None,
        seed: int = SEED, keep_models: bool = False,
        verbose: bool = True) -> Dict[str, object]:
    """Run AG-XGBoost (or one of its ablation variants).

    Parameters
    ----------
    method : 'agxgb' (location-adaptive bandwidth) or 'gxgb' (single bandwidth
        for the whole study area, the G-XGBoost baseline).
    search : 'coarse_golden' (two-stage search) or 'step' (stepwise search).
    borrow : if True, tune at m representative locations and propagate the
        configuration to the nearest representative location; if False, one
        global configuration is used for all locations.
    """
    grid = grid or LOCAL_GRID
    base = base or BASE_PARAMS
    n = len(y)
    ctx = build_context(coords)
    lo, hi = BW_MIN, n - 1
    t0 = time.time()

    # --- hyper-parameter configuration -------------------------------------
    if borrow:
        m = max(5, int(round(m_frac * n)))
        anchors = farthest_point_anchors(coords, m)
        if verbose:
            print(f"[1/3] local tuning at {len(anchors)} representative locations "
                  f"(m = max(5, {m_frac:.0%} N)) ...", flush=True)
        local_params: Dict[int, Dict[str, object]] = {}
        for a in anchors:
            # proxy bandwidth of the representative location
            proxy = LocalFitter(X, y, ctx, base, seed=seed)
            criterion = (lambda b, _a=int(a), _f=proxy: _f.error(_a, b))
            res = (coarse_golden_search(criterion, lo, hi, n_anchors)
                   if search == "coarse_golden" else step_search(criterion, lo, hi))
            local_params[a] = tune_local(X, y, ctx, a, res.bw, grid, base, seed)
        # propagate to the nearest representative location
        adist = distance_matrix(coords, coords[anchors])
        nearest = anchors[np.argmin(adist, axis=1)]
        params_of = {i: local_params[int(nearest[i])] for i in range(n)}
    else:
        if verbose:
            print("[1/3] global hyper-parameter search ...", flush=True)
        gparams = tune_global(X, y, GLOBAL_GRID, base, seed)
        params_of = {i: dict(gparams) for i in range(n)}
        anchors = np.array([], dtype=int)

    # --- bandwidth selection ------------------------------------------------
    if verbose:
        print("[2/3] bandwidth selection ...", flush=True)
    bws = np.zeros(n, dtype=int)
    n_evals = 0
    if method == "gxgb":
        # one shared bandwidth, chosen on a subset of locations (as in G-XGBoost)
        sample = np.unique(np.linspace(0, n - 1, min(n, 60)).astype(int))
        fitter = LocalFitter(X, y, ctx, params_of[0], seed=seed)

        def shared(b: int) -> float:
            return float(np.mean([fitter.error(int(i), int(b)) for i in sample]))

        res = (step_search(shared, lo, hi) if search == "step"
               else coarse_golden_search(shared, lo, hi, n_anchors))
        bws[:] = res.bw
        n_evals = res.n_evals
    else:
        for i in range(n):
            fitter = LocalFitter(X, y, ctx, params_of[i], seed=seed)
            criterion = (lambda b, _i=int(i), _f=fitter: _f.error(_i, b))
            res = (coarse_golden_search(criterion, lo, hi, n_anchors)
                   if search == "coarse_golden" else step_search(criterion, lo, hi))
            bws[i] = res.bw
            n_evals += res.n_evals
        # a fresh fitter is used per location because the hyper-parameters
        # are location specific

    # --- final models and leave-one-out predictions -------------------------
    if verbose:
        print("[3/3] fitting the final local models ...", flush=True)
    preds = np.zeros(n)
    models: List[Optional[XGBRegressor]] = [None] * n
    for i in range(n):
        fitter = LocalFitter(X, y, ctx, params_of[i], seed=seed, keep_models=keep_models)
        model, pred = fitter.predict(i, int(bws[i]))
        models[i] = model
        preds[i] = pred

    elapsed = time.time() - t0
    out = {
        "method": method,
        "search": search,
        "borrow": bool(borrow),
        "n_locations": n,
        "n_fits": int(n_evals),
        "bandwidth": bws,
        "prediction": preds,
        "metrics": metrics(y, preds),
        "elapsed_sec": float(elapsed),
        "anchors": anchors,
        "params_of": params_of,
        "models": models,
    }
    if verbose:
        m = out["metrics"]
        print(f"      R2 = {m['r2']:.4f} | MAE = {m['mae']:.4f} | "
              f"RMSE = {m['rmse']:.4f} | {elapsed:.1f} s")
    return out


def shap_attributions(result: Dict[str, object], X: np.ndarray,
                      feature_names: Sequence[str]) -> Optional[pd.DataFrame]:
    """Per-location SHAP summary (mean absolute value per feature).

    Returns None when the optional dependency ``shap`` is not installed or
    when the final models were not kept.
    """
    if shap is None:
        print("      [skip] shap is not installed (pip install shap)", flush=True)
        return None
    models = result.get("models") or []
    rows: List[np.ndarray] = []
    for i, model in enumerate(models):
        if model is None:
            continue
        try:
            values = shap.TreeExplainer(model).shap_values(X)
        except Exception:  # pragma: no cover - depends on the xgboost/shap versions
            return None
        rows.append(np.abs(np.asarray(values, dtype=float)).mean(axis=0))
    if not rows:
        return None
    matrix = np.vstack(rows)
    return pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": matrix.mean(axis=0),
        "share_pct": 100.0 * matrix.mean(axis=0) / max(matrix.mean(axis=0).sum(), 1e-12),
    }).sort_values("mean_abs_shap", ascending=False)


# ----------------------------------------------------------------------------
# 7. Data loading and a synthetic example
# ----------------------------------------------------------------------------
def load_csv(path: str, x_cols: Sequence[str], y_col: str,
             coord_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Read a CSV file with feature, target and coordinate columns."""
    df = pd.read_csv(path)
    missing = [c for c in list(x_cols) + [y_col] + list(coord_cols) if c not in df.columns]
    if missing:
        raise SystemExit(f"columns not found in {path}: {missing}")
    df = df[list(x_cols) + [y_col] + list(coord_cols)].dropna()
    X = df[list(x_cols)].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    coords = df[list(coord_cols)].to_numpy(dtype=float)
    return X, y, coords, list(x_cols)


def make_demo(n: int = 240, p: int = 5, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Synthetic spatially varying regression problem (test case).

    Cluster centres are drawn uniformly in the unit square; each cluster has
    its own regression coefficients, so the response surface is spatially
    non-stationary and nonlinear (two-way interactions are added).
    """
    rng = np.random.default_rng(seed)
    n_clusters = 6
    centres = rng.uniform(0.2, 0.8, size=(n_clusters, 2))
    coords = rng.uniform(0.0, 1.0, size=(n, 2))
    labels = np.argmin(((coords[:, None, :] - centres[None, :, :]) ** 2).sum(-1), axis=1)
    X = rng.normal(size=(n, p))
    betas = rng.normal(size=(n_clusters, p))
    betas[rng.random((n_clusters, p)) < 0.3] = 0.0
    beta0 = rng.normal(size=n_clusters)
    y = (X * betas[labels]).sum(1) + beta0[labels] \
        + 0.6 * X[:, 0] * X[:, 1] + rng.normal(scale=0.3, size=n)
    names = [f"x{j + 1}" for j in range(p)]
    return X, y, coords, names


# ----------------------------------------------------------------------------
# 8. Command line interface
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AG-XGBoost: location-adaptive bandwidth geographically "
                    "weighted XGBoost (single-file reference implementation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_argument_group("input data")
    src.add_argument("--data", help="CSV file with features, target and coordinates")
    src.add_argument("--x", nargs="+", help="names of the feature columns")
    src.add_argument("--y", help="name of the target column")
    src.add_argument("--coords", nargs=2, help="names of the two coordinate columns")
    src.add_argument("--demo", action="store_true", help="run the synthetic example")
    src.add_argument("--demo-n", type=int, default=150,
                     help="number of observations of the synthetic example")

    alg = p.add_argument_group("algorithm")
    alg.add_argument("--method", choices=["agxgb", "gxgb"], default="agxgb",
                     help="agxgb: location-adaptive bandwidth; gxgb: single bandwidth")
    alg.add_argument("--search", choices=["coarse_golden", "step"],
                     default="coarse_golden", help="bandwidth search strategy")
    alg.add_argument("--no-borrow", action="store_true",
                     help="disable spatial hyper-parameter borrowing")
    alg.add_argument("--anchors", type=int, default=N_ANCHORS,
                     help="number of anchors of the coarse scan")
    alg.add_argument("--m-frac", type=float, default=M_FRACTION,
                     help="fraction of representative locations")
    alg.add_argument("--fast", action="store_true",
                     help="small hyper-parameter grid (quick run)")
    alg.add_argument("--seed", type=int, default=SEED)

    out = p.add_argument_group("output")
    out.add_argument("--out", default="./agxgboost_out", help="output directory")
    out.add_argument("--no-shap", action="store_true", help="skip SHAP attribution")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.demo:
        X, y, coords, names = make_demo(n=args.demo_n)
        print(f"synthetic data: n = {len(y)}, features = {len(names)}")
    elif args.data:
        if not (args.x and args.y and args.coords):
            raise SystemExit("--x, --y and --coords are required with --data")
        X, y, coords, names = load_csv(args.data, args.x, args.y, args.coords)
        print(f"loaded {args.data}: n = {len(y)}, features = {len(names)}")
    else:
        raise SystemExit("provide --data ... or --demo")

    grid = SMALL_GRID if args.fast else LOCAL_GRID
    base = FAST_PARAMS if args.fast else BASE_PARAMS
    result = run(X, y, coords, method=args.method, search=args.search,
                 borrow=not args.no_borrow, n_anchors=args.anchors,
                 m_frac=args.m_frac, grid=grid, base=base, seed=args.seed,
                 keep_models=not args.no_shap)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": result["method"],
        "search": result["search"],
        "borrow": result["borrow"],
        "n_locations": result["n_locations"],
        "n_fits": result["n_fits"],
        "elapsed_sec": round(result["elapsed_sec"], 2),
        "metrics": result["metrics"],
        "env": {"python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "xgboost": xgboost.__version__},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    per_loc = pd.DataFrame({
        "location": np.arange(result["n_locations"]),
        "x": coords[:, 0], "y": coords[:, 1],
        "bandwidth": result["bandwidth"],
        "y_true": y, "y_pred": result["prediction"],
        "sq_error": (y - result["prediction"]) ** 2,
    })
    per_loc.to_csv(out / "per_location.csv", index=False)

    if not args.no_shap:
        table = shap_attributions(result, X, names)
        if table is not None:
            table.to_csv(out / "shap_summary.csv", index=False)
            print("\nmean |SHAP| by feature:")
            print(table.to_string(index=False))

    print(f"\noutputs written to {out.resolve()}")
    print(f"  metrics.json, per_location.csv"
          + ("" if args.no_shap else ", shap_summary.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
