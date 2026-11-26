"""Stream RVL-CDIP (test split), keep a few documents per class that DiT classifies correctly."""
import json, sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/rvlcdip_samples")
WANT = {"invoice": 3, "letter": 2, "form": 2, "memo": 2, "resume": 2, "budget": 2}

OUT.mkdir(parents=True, exist_ok=True)
ds = load_dataset("chainyo/rvl-cdip", split="test", streaming=True)
names = ds.features["label"].names
proc = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip").eval()
id2label = model.config.id2label  # DiT's own label order

kept, meta = {k: 0 for k in WANT}, []
BLOCK = 2500  # the test split is ordered by class: 2500 pages each (alphabetical class order)
for cls in WANT:
    start = names.index(cls) * BLOCK
    for ex in ds.skip(start).take(BLOCK):
        if kept[cls] >= WANT[cls]:
            break
        if names[ex["label"]] != cls:
            continue
        img = ex["image"].convert("RGB")
        with torch.no_grad():
            probs = model(**proc(img, return_tensors="pt")).logits.softmax(-1)[0]
        pred = id2label[int(probs.argmax())]
        if pred != cls:
            continue  # keep only correctly classified documents
        kept[cls] += 1
        fn = f"{cls.replace(' ', '_')}_{kept[cls]}.png"
        img.save(OUT / fn)
        meta.append({"file": fn, "label": cls, "pred": pred, "prob": round(float(probs.max()), 4)})
        print(fn, pred, round(float(probs.max()), 3), flush=True)

json.dump(meta, open(OUT / "samples.json", "w"), indent=1)
print("kept", kept)
