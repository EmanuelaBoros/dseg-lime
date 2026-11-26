# DSEG-LIME on RVL-CDIP

Tests the [DSEG-LIME](https://github.com/patrick-knab/DSEG-LIME) explainer (Knab et al., MIT) on RVL-CDIP document pages, with two segmenters:
**SAM** (the original) and **OCR** blocks/lines (PaddleOCR, or Tesseract) plugged into the same code. The classifier being explained is
DiT or ResNet-101, both fine-tuned on RVL-CDIP.

| Path | What |
|---|---|
| `notebooks/dseg_lime_rvlcdip_dit.ipynb` | DSEG-LIME explaining DiT (`microsoft/dit-base-finetuned-rvlcdip`) |
| `notebooks/dseg_lime_rvlcdip_resnet.ipynb` | the same, explaining ResNet-101 (`bdpc/resnet101_rvl-cdip`) |
| `code/ocr_masks.py` | OCR boxes as a drop-in replacement for SAM masks in DSEG-LIME |
| `code/dseg_shims.py` | stubs DSEG-LIME's TensorFlow imports so the upstream code runs unmodified |
| `code/fetch_rvlcdip_samples.py` | pulls a few correctly classified RVL-CDIP test pages into `data/rvlcdip_samples/` |
| `code/build_dseg_notebook.py` | generates both notebooks (`dit` or `resnet`); edit this, not the `.ipynb` files |

## Setup

The OCR section uses PaddleOCR (`pip install paddlepaddle paddleocr`); the optional Tesseract backend needs the `tesseract` binary (`brew install tesseract`) and `pytesseract`.

```sh
git clone https://github.com/patrick-knab/DSEG-LIME baselines/DSEG-LIME   # upstream, not vendored here
python -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision transformers datasets segment-anything networkx seaborn scikit-learn scipy opencv-python pandas paddlepaddle paddleocr ipykernel
mkdir -p models && curl -L -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
.venv/bin/python code/fetch_rvlcdip_samples.py
.venv/bin/python code/build_dseg_notebook.py dit && .venv/bin/python code/build_dseg_notebook.py resnet
```

Then open a notebook. The pages used are the `DOCS` list in its setup cell.

## Credits

DSEG-LIME: P. Knab, S. Marton, C. Bartelt, *Beyond Pixels: Enhancing LIME with Hierarchical Features and Segmentation Foundation Models* (arXiv:2403.07733), <https://github.com/patrick-knab/DSEG-LIME> (MIT).
