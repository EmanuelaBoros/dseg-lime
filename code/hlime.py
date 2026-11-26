"""Hierarchical LIME over nested segmentations (blocks -> lines -> words).

Level 0 runs plain LIME on the coarsest segmentation. At each finer level,
only the children of the segments selected at the previous level are
perturbed (everything else stays visible), and the children's coefficients
are constrained to sum to their parent's coefficient.  Children of segments
that were not selected inherit their parent's coefficient, split by area.

The explainer is model-agnostic: it only needs
    predict_fn(images: np.ndarray[n, H, W, C]) -> np.ndarray[n]
returning the score of the class being explained.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class LevelResult:
    coef: np.ndarray        # (n_segments,) coefficient for every segment id of the level
    perturbed: np.ndarray   # (n_segments,) bool: True if fitted at this level, False if inherited
    intercept: float
    n_samples: int
    selected: np.ndarray = None  # (n_segments,) bool: kept for refinement at the next level


def parent_map(fine, coarse):
    """parent[i] = id of the coarse segment overlapping fine segment i the most."""
    n_f, n_c = fine.max() + 1, coarse.max() + 1
    counts = np.bincount((fine * n_c + coarse).ravel(), minlength=n_f * n_c)
    return counts.reshape(n_f, n_c).argmax(1)


def kernel_weights(z, kernel_width):
    removed = 1.0 - z.mean(1)  # fraction of units masked
    return np.exp(-(removed ** 2) / kernel_width ** 2)


def constrained_ridge(z, y, sw, alpha, groups=None, targets=None):
    """Weighted ridge with a free intercept and optional equality constraints.

    groups: list of index arrays; for each, sum(w[idx]) == targets[k].
    Solved exactly through the KKT system.
    """
    n, d = z.shape
    X = np.hstack([np.ones((n, 1)), z])
    P = np.diag([0.0] + [alpha] * d)
    H = 2 * (X.T @ (sw[:, None] * X) + P)
    g = 2 * X.T @ (sw * y)
    if groups:
        A = np.zeros((len(groups), d + 1))
        for k, idx in enumerate(groups):
            A[k, 1 + np.asarray(idx)] = 1.0
        K = np.block([[H, A.T], [A, np.zeros((len(groups), len(groups)))]])
        rhs = np.concatenate([g, np.asarray(targets, float)])
    else:
        K, rhs = H, g
    sol = np.linalg.solve(K, rhs)
    return sol[0], sol[1:d + 1]


def _sample_masks(n_samples, d, rng):
    z = rng.integers(0, 2, size=(n_samples, d)).astype(float)
    z[0] = 1.0  # the original image
    return z


def _render(image, seg, units, z, fill):
    """Mask the segments of `units` whose z is 0; all other pixels stay."""
    out = np.empty((len(z),) + image.shape, dtype=image.dtype)
    max_id = seg.max() + 1
    for i, row in enumerate(z):
        keep = np.ones(max_id, bool)
        keep[units[row == 0]] = False
        img = image.copy()
        img[~keep[seg]] = fill
        out[i] = img
    return out


def hierarchical_lime(image, segmentations, predict_fn, n_samples=500,
                      top_k=3, alpha=1.0, kernel_width=0.5, fill=1.0,
                      constrained=True, seed=0):
    """Run hierarchical LIME.

    segmentations: list of (H, W) int maps with ids 0..n-1, coarse -> fine.
                   Each level must be (approximately) nested in the previous one.
    n_samples, top_k: int, or one value per level (top_k is ignored on the last level).
    Returns a list of LevelResult, one per level.
    """
    rng = np.random.default_rng(seed)
    L = len(segmentations)
    n_samples = [n_samples] * L if np.isscalar(n_samples) else list(n_samples)
    top_k = [top_k] * L if np.isscalar(top_k) else list(top_k)

    results = []
    for lvl, seg in enumerate(segmentations):
        n_seg = seg.max() + 1
        coef = np.zeros(n_seg)
        perturbed = np.zeros(n_seg, bool)

        if lvl == 0:
            units = np.arange(n_seg)
            groups = targets = None
        else:
            prev, parents = results[-1], parent_map(seg, segmentations[lvl - 1])
            selected = np.flatnonzero(prev.perturbed & prev.selected)
            units = np.flatnonzero(np.isin(parents, selected))
            # children of unselected parents inherit, split by area
            area = np.bincount(seg.ravel(), minlength=n_seg).astype(float)
            parent_area = np.bincount(segmentations[lvl - 1].ravel(),
                                      minlength=prev.coef.size).astype(float)
            coef = prev.coef[parents] * area / parent_area[parents]
            pos = {u: i for i, u in enumerate(units)}
            groups = [[pos[u] for u in units if parents[u] == p] for p in selected]
            targets = [prev.coef[p] for p in selected]

        z = _sample_masks(n_samples[lvl], len(units), rng)
        y = np.asarray(predict_fn(_render(image, seg, units, z, fill)), float)
        sw = kernel_weights(z, kernel_width)
        b0, w = constrained_ridge(z, y, sw, alpha,
                                  groups if constrained else None,
                                  targets if constrained else None)
        coef[units] = w
        perturbed[units] = True

        res = LevelResult(coef, perturbed, b0, n_samples[lvl])
        if lvl < L - 1:
            k = min(top_k[lvl], len(units))
            sel = np.zeros(n_seg, bool)
            sel[units[np.argsort(-np.abs(w))[:k]]] = True
            res.selected = sel
        results.append(res)
    return results
