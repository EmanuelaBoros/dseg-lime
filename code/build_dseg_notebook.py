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
from PIL import Image
from skimage.segmentation import mark_boundaries
from transformers import AutoImageProcessor, AutoModelForImageClassification
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from Utilities.dseg_lime import DSEG_Lime

SAMPLES = ROOT / "data" / "rvlcdip_samples"
MAX_SIDE = 512        # documents are downscaled for SAM / LIME speed
NUM_SAMPLES = 256     # LIME neighbourhood size per iteration
ITERATIONS = 2        # DSEG hierarchy depth (coarse -> fine)
DEVICE = "cpu"        # SAM + DiT; set "mps"/"cuda" if available and stable""")

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

md("## 3. One document, step by step")
code("""def load_doc(name):
    path = SAMPLES / name
    img = Image.open(path).convert("RGB")
    s = MAX_SIDE / max(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)))
    resized = SAMPLES / "_resized" / name
    resized.parent.mkdir(exist_ok=True)
    img.save(resized)                       # DSEG wants an image_path alongside the array
    return np.array(img), str(resized)

name = "invoice_1.png"
image, path = load_doc(name)
probs = classifier_fn(image[None])[0]
top = probs.argsort()[::-1][:3]
print([(id2label[int(i)], round(float(probs[i]), 3)) for i in top])
plt.imshow(image); plt.axis("off"); plt.show()""")

code("""t0 = time.time()
explainer = DSEG_Lime()
exp = explainer.explain_instance(image=image, classifier_fn=classifier_fn,
                                 feature_extractor=mask_generator, model=sam, image_path=path,
                                 num_samples=NUM_SAMPLES, iterations=ITERATIONS, progress_bar=False)
print(f"{time.time() - t0:.0f}s, {len(np.unique(exp.segments))} segments at the last level")""")

md("""## 4. Visualise: segments ranked by LIME coefficient for the predicted class

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
ax[0].imshow(image); ax[0].set_title("input"); ax[0].axis("off")
show(exp, image, label, ax=ax[1]); ax[1].set_title(f"DSEG-LIME: {id2label[int(label)]}")
plt.show()""")

md("## 5. All sample documents")
code("""rows = []
files = sorted(p.name for p in SAMPLES.glob("*.png"))
fig, axes = plt.subplots(2, (len(files) + 1) // 2, figsize=(3.2 * ((len(files) + 1) // 2), 9))
for ax, f in zip(axes.ravel(), files):
    img, p = load_doc(f)
    t0 = time.time()
    e = DSEG_Lime().explain_instance(image=img, classifier_fn=classifier_fn, feature_extractor=mask_generator,
                                     model=sam, image_path=p, num_samples=NUM_SAMPLES,
                                     iterations=ITERATIONS, progress_bar=False)
    lab = e.top_labels[0]
    show(e, img, lab, ax=ax); ax.set_title(f"{f[:-4]}\\n→ {id2label[int(lab)]}", fontsize=8)
    rows.append({"file": f, "pred": id2label[int(lab)], "n_segments": int(len(np.unique(e.segments))),
                 "local_R2": round(float(e.score[lab]), 3), "seconds": round(time.time() - t0)})
plt.tight_layout(); plt.show()
import pandas as pd; pd.DataFrame(rows)""")

md("""## Notes for the comparison with hierarchical LIME

* DSEG-LIME builds its hierarchy from SAM masks (containment graph), not from document layout; segments are not semantic blocks.
* `local_R2` is LIME's own weighted R² of the surrogate (a local-fidelity proxy, see Exp. 1 of the paper).
* SAM yields only ~6-8 usable masks on a 512px invoice (after DSEG's `remove_small_clusters`), so the baseline's
  segments are coarse blobs, not layout blocks or words. That is the gap hierarchical semantic-block LIME targets.
* Hide colour: the baseline default (mean colour per segment) is kept, rather than white as in hierarchical LIME.""")

nbf.write(nb, "notebooks/dseg_lime_rvlcdip.ipynb")
