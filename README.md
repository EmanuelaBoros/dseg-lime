# Hierarchical attribution over semantic blocks

Code for hierarchical attribution over semantic blocks for explainable document classification, plus a DSEG-LIME baseline on RVL-CDIP.

| Path | What |
|---|---|
| `code/hlime.py`, `code/test_hlime.py` | Hierarchical LIME core (model-agnostic), with a toy sanity test |
| `notebooks/dseg_lime_rvlcdip.ipynb` | DSEG-LIME baseline explaining DiT on RVL-CDIP pages |
| `code/fetch_rvlcdip_samples.py` | Pulls a few correctly-classified RVL-CDIP test pages into `data/rvlcdip_samples/` |
| `code/dseg_shims.py` | Stubs DSEG-LIME's TensorFlow imports so the baseline runs unmodified |

## Running the DSEG-LIME baseline

```sh
git clone https://github.com/patrick-knab/DSEG-LIME baselines/DSEG-LIME   # upstream, MIT, (c) Patrick Knab
python -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision transformers datasets segment-anything networkx seaborn scikit-learn scipy opencv-python pandas ipykernel
mkdir -p models && curl -L -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
.venv/bin/python code/fetch_rvlcdip_samples.py
```

Then open `notebooks/dseg_lime_rvlcdip.ipynb`.

## Credits

DSEG-LIME: Knab et al., <https://github.com/patrick-knab/DSEG-LIME> (MIT). It is not vendored here.
