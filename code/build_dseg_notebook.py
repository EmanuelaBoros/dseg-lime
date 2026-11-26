import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))

md("""# DSEG-LIME baseline on RVL-CDIP documents

Runs the [DSEG-LIME](https://github.com/patrick-knab/DSEG-LIME) baseline (SAM-based, hierarchical segmentation + LIME)
to explain a **DiT** document classifier (`microsoft/dit-base-finetuned-rvlcdip`) on a few RVL-CDIP test pages.

* Samples: `data/rvlcdip_samples/` (made by `code/fetch_rvlcdip_samples.py`, only pages DiT classifies correctly).
* Baseline code is used unmodified from `baselines/DSEG-LIME`; TensorFlow imports are stubbed by `code/dseg_shims.py`.
* Segmenter: SAM ViT-B (the repo's demo uses ViT-H; ViT-B is ~6x smaller and fast enough on CPU).""")

code("""import sys, time, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "baselines" / "DSEG-LIME")]

import dseg_shims; dseg_shims.install()          # must come before importing Utilities.dseg_lime

import numpy as np, torch, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
from skimage.segmentation import mark_boundaries
from transformers import AutoImageProcessor, AutoModelForImageClassification
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from Utilities.dseg_lime import DSEG_Lime
from ocr_masks import OCRMaskGenerator          # OCR layout boxes as a drop-in for SAM

SAMPLES = ROOT / "data" / "rvlcdip_samples"
MAX_SIDE = 512        # documents are downscaled for SAM / LIME speed
NUM_SAMPLES = 256     # LIME neighbourhood size per iteration
ITERATIONS = 2        # DSEG hierarchy depth (coarse -> fine)
DEVICE = "cpu"        # SAM + DiT; set "mps"/"cuda" if available and stable

import contextlib
@contextlib.contextmanager
def dseg_plots(title=None):
    \"\"\"DSEG-LIME calls plt.imshow(segments); plt.show(); plt.close() once per hierarchy level, untitled, on matplotlib's
    *current* axes. title=None silences that (needed inside our subplot grids, where it would draw over a panel and
    close the figure). With a title, the per-level segmentation plots are kept and titled "<title>, level N".\"\"\"
    real = plt.show, plt.imshow, plt.close
    n = [0]
    if title is None:
        plt.imshow = plt.show = plt.close = lambda *a, **k: None
    else:
        def show(*a, **k):
            n[0] += 1
            plt.title(f"{title}, level {n[0]}", fontsize=9)
            real[0](*a, **k)
        plt.show = show
    try:
        yield
    finally:
        plt.show, plt.imshow, plt.close = real""")

md("## 1. Classifier to explain: DiT fine-tuned on RVL-CDIP")
code("""proc = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
dit = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip").eval().to(DEVICE)
id2label = dit.config.id2label

@torch.no_grad()
def classifier_fn(images):
    \"\"\"DSEG/LIME interface: (n, H, W, 3) uint8 -> (n, 16) class probabilities.\"\"\"
    x = proc([Image.fromarray(np.asarray(i, dtype=np.uint8)) for i in images], return_tensors="pt")
    return dit(**x.to(DEVICE)).logits.softmax(-1).cpu().numpy()""")

md("## 2. SAM segmenter")
code("""ckpt = ROOT / "models" / "sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=str(ckpt)).to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=512, points_per_side=32)   # 32 = the repo demo's default""")

code("""def load_doc(name):
    path = SAMPLES / name
    img = Image.open(path).convert("RGB")
    s = MAX_SIDE / max(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)))
    resized = SAMPLES / "_resized" / name
    resized.parent.mkdir(exist_ok=True)
    img.save(resized)                       # DSEG wants an image_path alongside the array
    return np.array(img), str(resized)""")

md("""## 2b. What SAM sees in a document

One page per class. Left: the page. Middle: **all raw SAM masks** (random colours, large masks underneath).
Right: the masks DSEG-LIME actually keeps, after `remove_small_clusters` (masks under 500 px are dropped).""")
code("""def overlay(img, masks, seed=0):
    rng = np.random.default_rng(seed)
    canvas = img.astype(float)
    for m in sorted(masks, key=lambda m: -m["area"]):       # big first, so small masks stay visible
        seg = m["segmentation"]
        canvas[seg] = 0.4 * canvas[seg] + 0.6 * rng.integers(40, 255, 3)
    return canvas.astype(np.uint8)

first_per_class = {}
for f in sorted(p.name for p in SAMPLES.glob("*.png")):
    first_per_class.setdefault(f.rsplit("_", 1)[0], f)

fig, axes = plt.subplots(len(first_per_class), 3, figsize=(9, 4.2 * len(first_per_class)))
for row, (cls, f) in tqdm(zip(axes, first_per_class.items()), total=len(first_per_class), desc="SAM masks"):
    img, _ = load_doc(f)
    raw = mask_generator.generate(img)
    kept, _ = DSEG_Lime().remove_small_clusters(raw, 500, plot=False)
    row[0].imshow(img); row[0].set_title(f"{f}\\ntrue class: {cls}", fontsize=8)
    row[1].imshow(overlay(img, raw)); row[1].set_title(f"SAM: all raw masks ({len(raw)})", fontsize=8)
    row[2].imshow(overlay(img, kept)); row[2].set_title(f"SAM: masks DSEG keeps (>= 500 px): {len(kept)}", fontsize=8)
    for a in row: a.axis("off")
plt.tight_layout(); plt.show()""")

md("""## 2c. The same pages segmented by OCR instead of SAM

`OCRMaskGenerator` (`code/ocr_masks.py`) turns OCR boxes into masks, with the same interface as SAM's mask generator.
DSEG-LIME builds its hierarchy from mask containment, so lines nest inside blocks.

* `BACKEND = "paddle"` (default): PaddleOCR, the OCR family used in the *When superpixels fail on documents* paper.
  It returns text **lines** only, so **blocks are a geometric heuristic** (lines that overlap horizontally and are vertically close are merged), not a layout model.
* `BACKEND = "tesseract"`: native blocks / lines / words.

Words are left out by default: DSEG drops masks under 500 px, which removes most single words at this resolution.""")
code("""BACKEND = "paddle"                # or "tesseract"
LEVELS = ("block", "line")        # "word" only works with BACKEND = "tesseract"
ocr_generator = OCRMaskGenerator(backend=BACKEND, levels=LEVELS)

fig, axes = plt.subplots(len(first_per_class), 3, figsize=(9, 4.2 * len(first_per_class)))
for row, (cls, f) in tqdm(zip(axes, first_per_class.items()), total=len(first_per_class), desc="OCR masks"):
    img, _ = load_doc(f)
    raw = ocr_generator.generate(img)
    kept, _ = DSEG_Lime().remove_small_clusters(raw, 500, plot=False)
    n_blocks = sum(m["level"] == "block" for m in raw)
    row[0].imshow(img); row[0].set_title(f"{f}\\ntrue class: {cls}", fontsize=8)
    row[1].imshow(overlay(img, raw)); row[1].set_title(f"OCR ({BACKEND}): {n_blocks} blocks + {len(raw) - n_blocks} lines/words", fontsize=8)
    row[2].imshow(overlay(img, kept)); row[2].set_title(f"OCR: masks DSEG keeps (>= 500 px): {len(kept)}", fontsize=8)
    for a in row: a.axis("off")
plt.tight_layout(); plt.show()""")

md("## 3. One document, step by step: SAM")
code("""name = "invoice_1.png"
image, path = load_doc(name)
probs = classifier_fn(image[None])[0]
top = probs.argsort()[::-1][:3]
print([(id2label[int(i)], round(float(probs[i]), 3)) for i in top])
plt.imshow(image); plt.axis("off")
plt.title(f"{name}\\ntrue class: {name.rsplit('_', 1)[0]} | DiT predicts: {id2label[int(top[0])]} ({probs[top[0]]:.2f})", fontsize=9)
plt.show()""")

code("""t0 = time.time()
explainer = DSEG_Lime()
with dseg_plots(f"{name} | DSEG segmentation (SAM)"):      # titles the per-level segmentation plots
    exp = explainer.explain_instance(image=image, classifier_fn=classifier_fn,
                                     feature_extractor=mask_generator, model=sam, image_path=path,
                                     num_samples=NUM_SAMPLES, iterations=ITERATIONS, progress_bar=False)
print(f"SAM: {time.time() - t0:.0f}s, {len(np.unique(exp.segments))} segments at the last level")""")

md("""## 4. Visualise (SAM): segments ranked by LIME coefficient for the predicted class

Green = pushes towards the predicted class, red = pushes against. Only the strongest segments are drawn.""")
code("""def show(exp, image, label, k=8, ax=None):
    ax = ax or plt.gca()
    coefs = {t[0]: t[1] for t in exp.local_exp[label]}   # entries are (segment_id, weight, 0.0)
    heat = np.zeros(exp.segments.shape)
    for sid, w in sorted(coefs.items(), key=lambda t: -abs(t[1]))[:k]:
        heat[exp.segments == sid] = w
    m = np.abs(heat).max() or 1
    ax.imshow(image); ax.imshow(heat, cmap="RdYlGn", vmin=-m, vmax=m, alpha=0.55)
    ax.imshow(mark_boundaries(np.zeros(image.shape[:2]), exp.segments.astype(int))[..., 0], cmap="gray_r", alpha=0.25)
    ax.axis("off")

label = exp.top_labels[0]
print("explained class:", id2label[int(label)])
fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].imshow(image); ax[0].set_title(f"input: {name}", fontsize=9); ax[0].axis("off")
show(exp, image, label, ax=ax[1])
ax[1].set_title(f"DSEG-LIME (SAM): 8 strongest segments for '{id2label[int(label)]}'\\ngreen = pushes towards the class, red = against", fontsize=9)
plt.show()""")

md("""## 3b. One document, step by step: OCR

The same page, classifier and settings as section 3, but the segmenter is `ocr_generator` from section 2c (`BACKEND`, PaddleOCR by default) instead of SAM.""")
code("""print([(id2label[int(i)], round(float(probs[i]), 3)) for i in top], "(same page, same classifier as section 3)")
plt.imshow(image); plt.axis("off")
plt.title(f"{name}\\ntrue class: {name.rsplit('_', 1)[0]} | DiT predicts: {id2label[int(top[0])]} ({probs[top[0]]:.2f})", fontsize=9)
plt.show()""")

code("""t0 = time.time()
with dseg_plots(f"{name} | DSEG segmentation (OCR: {BACKEND})"):
    exp_ocr = DSEG_Lime().explain_instance(image=image, classifier_fn=classifier_fn,
                                           feature_extractor=ocr_generator, model=None, image_path=path,
                                           num_samples=NUM_SAMPLES, iterations=ITERATIONS, progress_bar=False)
print(f"OCR ({BACKEND}): {time.time() - t0:.0f}s, {len(np.unique(exp_ocr.segments))} segments at the last level")""")

md("## 4b. Visualise (OCR): segments ranked by LIME coefficient for the predicted class")
code("""label_ocr = exp_ocr.top_labels[0]
print("explained class:", id2label[int(label_ocr)])
fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].imshow(image); ax[0].set_title(f"input: {name}", fontsize=9); ax[0].axis("off")
show(exp_ocr, image, label_ocr, ax=ax[1])
ax[1].set_title(f"DSEG-LIME (OCR, {BACKEND}): 8 strongest segments for '{id2label[int(label_ocr)]}'\\ngreen = pushes towards the class, red = against", fontsize=9)
plt.show()""")

md("## 4c. Side by side: SAM vs OCR on this page")
code("""fig, ax = plt.subplots(1, 3, figsize=(15, 7))
ax[0].imshow(image); ax[0].set_title(f"input: {name}", fontsize=9); ax[0].axis("off")
show(exp, image, label, ax=ax[1])
ax[1].set_title(f"SAM ({len(np.unique(exp.segments))} segments) -> '{id2label[int(label)]}'", fontsize=9)
show(exp_ocr, image, label_ocr, ax=ax[2])
ax[2].set_title(f"OCR, {BACKEND} ({len(np.unique(exp_ocr.segments))} segments) -> '{id2label[int(label_ocr)]}'", fontsize=9)
plt.tight_layout(); plt.show()""")

md("## 5. All sample documents")
code("""rows = []
files = sorted(p.name for p in SAMPLES.glob("*.png"))
fig, axes = plt.subplots(2, (len(files) + 1) // 2, figsize=(3.2 * ((len(files) + 1) // 2), 9))
for ax, f in tqdm(zip(axes.ravel(), files), total=len(files), desc="DSEG-LIME (SAM)"):
    img, p = load_doc(f)
    t0 = time.time()
    with dseg_plots(None):
        e = DSEG_Lime().explain_instance(image=img, classifier_fn=classifier_fn, feature_extractor=mask_generator,
                                         model=sam, image_path=p, num_samples=NUM_SAMPLES,
                                         iterations=ITERATIONS, progress_bar=False)
    lab = e.top_labels[0]
    show(e, img, lab, ax=ax); ax.set_title(f"{f} (true: {f.rsplit('_', 1)[0]})\\nDSEG + SAM -> predicted: {id2label[int(lab)]}", fontsize=8)
    rows.append({"file": f, "pred": id2label[int(lab)], "n_segments": int(len(np.unique(e.segments))),
                 "local_R2": round(float(e.score[lab]), 3), "seconds": round(time.time() - t0)})
plt.tight_layout(); plt.show()
import pandas as pd; pd.DataFrame(rows)""")

md("""## 6. SAM vs OCR segments inside DSEG-LIME

Same classifier, same DSEG-LIME code, same settings; only the segmenter changes. One page per class.
`prob_drop_top3` is a quick deletion test: hide the 3 segments with the highest positive coefficient (white) and report how much the
predicted-class probability falls. A bigger drop means the explanation pointed at regions the model really relies on.""")
code("""def prob_drop_top3(e, img, label, k=3):
    pos = sorted(((t[0], t[1]) for t in e.local_exp[label] if t[1] > 0), key=lambda t: -t[1])[:k]
    hidden = img.copy()
    for sid, _ in pos:
        hidden[e.segments == sid] = 255
    p0, p1 = classifier_fn(np.stack([img, hidden]))[:, label]
    return float(p0 - p1)

cmp_rows = []
fig, axes = plt.subplots(len(first_per_class), 3, figsize=(9, 4.2 * len(first_per_class)))
for row, (cls, f) in tqdm(zip(axes, first_per_class.items()), total=len(first_per_class), desc="SAM vs OCR in DSEG"):
    img, p = load_doc(f)
    row[0].imshow(img); row[0].set_title(f"{f}\\ntrue class: {cls}", fontsize=8); row[0].axis("off")
    for ax, (name, gen, mdl) in zip(row[1:], [("SAM", mask_generator, sam), ("OCR", ocr_generator, None)]):
        t0 = time.time()
        with dseg_plots(None):
            e = DSEG_Lime().explain_instance(image=img, classifier_fn=classifier_fn, feature_extractor=gen, model=mdl,
                                             image_path=p, num_samples=NUM_SAMPLES, iterations=ITERATIONS, progress_bar=False)
        lab = e.top_labels[0]
        show(e, img, lab, ax=ax); ax.set_title(f"DSEG + {name} segments\\n-> predicted: {id2label[int(lab)]}", fontsize=8)
        cmp_rows.append({"doc": f, "segmenter": name, "pred": id2label[int(lab)],
                         "n_segments": int(len(np.unique(e.segments))), "local_R2": round(float(e.score[lab]), 3),
                         "prob_drop_top3": round(prob_drop_top3(e, img, lab), 3), "seconds": round(time.time() - t0)})
plt.tight_layout(); plt.show()
import pandas as pd
df = pd.DataFrame(cmp_rows); display(df)
df.groupby("segmenter")[["n_segments", "local_R2", "prob_drop_top3", "seconds"]].mean().round(3)""")

md("""## Notes for the comparison with hierarchical LIME

* DSEG-LIME builds its hierarchy from SAM masks (containment graph), not from document layout; segments are not semantic blocks.
* `local_R2` is LIME's own weighted R² of the surrogate (a local-fidelity proxy, see Exp. 1 of the paper).
* SAM yields only ~6-8 usable masks on a 512px invoice (after DSEG's `remove_small_clusters`), so the baseline's
  segments are coarse blobs, not layout blocks or words. That is the gap hierarchical semantic-block LIME targets.
* Hide colour: the baseline default (mean colour per segment) is kept, rather than white as in hierarchical LIME.""")

nbf.write(nb, "notebooks/dseg_lime_rvlcdip.ipynb")
