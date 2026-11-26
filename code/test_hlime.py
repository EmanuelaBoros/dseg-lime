"""Sanity test on a toy additive model: score = fraction of black pixels in a 20x20 region."""
import numpy as np

from hlime import hierarchical_lime


def grid(h, w, cell):
    ys, xs = np.mgrid[:h, :w]
    return (ys // cell) * (w // cell) + (xs // cell)


image = np.ones((100, 100))
image[:20, :20] = 0.0  # black patch inside the top-left 50x50 quadrant


def predict_fn(batch):
    return (batch[:, :20, :20] == 0).mean(axis=(1, 2))


# nested grids: 50px (4 blocks) -> 10px (100 cells) -> 5px (400 cells)
# cell ids differ between levels, so renumbering is implicit through parent_map
segs = [grid(100, 100, 50), grid(100, 100, 10), grid(100, 100, 5)]
res = hierarchical_lime(image, segs, predict_fn, n_samples=[200, 400, 400],
                        top_k=[1, 4], alpha=1e-3)

print("level 0 coef:", np.round(res[0].coef, 3))
assert res[0].coef.argmax() == 0, "top-left block should dominate"

# constraint: children of block 0 sum to its coefficient
kids = np.flatnonzero(res[1].perturbed)
print("level 1 sum over children of selected block:", res[1].coef[kids].sum().round(4),
      "vs parent", res[0].coef[0].round(4))
assert np.isclose(res[1].coef[kids].sum(), res[0].coef[0])

# the 4 cells covering the 20x20 patch (ids 0,1,10,11) should be top at level 1
top = set(np.argsort(-np.abs(res[1].coef))[:4])
print("level 1 top-4:", sorted(top))
assert top == {0, 1, 10, 11}

# level 2: the 4 selected 10px cells are refined into 5px cells, sums preserved
from hlime import parent_map
par = parent_map(segs[2], segs[1])
for p in np.flatnonzero(res[1].selected):
    kids2 = np.flatnonzero(par == p)
    assert np.isclose(res[2].coef[kids2].sum(), res[1].coef[p]), p
print("level 2 constraints OK")

# inherited values keep the total mass
assert np.isclose(res[1].coef.sum(), res[0].coef.sum())
print("OK")
