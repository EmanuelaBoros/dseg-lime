"""Drop-in replacement for SamAutomaticMaskGenerator that segments a page with OCR layout boxes.

DSEG-LIME only reads `mask["segmentation"]` (a bool HxW array) from `feature_extractor.generate(image)`
and builds its hierarchy from mask containment, so OCR blocks > lines (> words) become a real hierarchy.

Backends
  "paddle"    PaddleOCR 3.x (as in Telnoff et al., "When superpixels fail on documents"). It only returns text
              LINES, so blocks are built by a geometric heuristic (see `group_lines`), not by a layout model.
  "tesseract" pytesseract: native block / line / word levels. Needs the `tesseract` binary.
"""
import os

import cv2
import numpy as np

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def group_lines(boxes, x_overlap=0.3, y_gap=0.8):
    """Heuristic 'blocks': merge text lines that overlap horizontally and are vertically close.

    Two lines are linked when their horizontal overlap is >= x_overlap of the narrower one and the vertical
    gap between them is <= y_gap x the smaller line height. Connected components become blocks.
    Returns a list of lists of line indices.
    """
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            ov = min(a[2], b[2]) - max(a[0], b[0])
            narrow = min(a[2] - a[0], b[2] - b[0])
            gap = max(a[1], b[1]) - min(a[3], b[3])
            h = min(a[3] - a[1], b[3] - b[1])
            if narrow > 0 and ov >= x_overlap * narrow and gap <= y_gap * h:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


class OCRMaskGenerator:
    _paddle = None  # PaddleOCR instance, shared (loading takes a few seconds)

    def __init__(self, backend="paddle", levels=("block", "line"), min_conf=30, ocr_scale=2.0, lang="en", pad=2):
        """levels: any of "block", "line", "word" ("word" needs backend="tesseract"). DSEG drops masks under
        500 px, so words only survive on large images. ocr_scale upsamples the page before OCR.
        min_conf is 0-100."""
        self.backend, self.levels, self.min_conf = backend, levels, min_conf
        self.ocr_scale, self.lang, self.pad = ocr_scale, lang, pad
        if "word" in levels and backend == "paddle":
            raise ValueError('"word" level needs backend="tesseract"')

    # ---- OCR backends: each returns (line_boxes, word_boxes, block_ids_per_line) -----------------------------
    def _run_paddle(self, big):
        if OCRMaskGenerator._paddle is None:
            from paddleocr import PaddleOCR
            OCRMaskGenerator._paddle = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                                                 use_textline_orientation=False, lang=self.lang)
        r = OCRMaskGenerator._paddle.predict(big)[0]
        lines = [tuple(map(float, b)) for b, t, s in zip(r["rec_boxes"], r["rec_texts"], r["rec_scores"])
                 if t.strip() and s * 100 >= self.min_conf]
        blocks = [None] * len(lines)
        for k, idx in enumerate(group_lines(lines)):
            for i in idx:
                blocks[i] = k
        return lines, [], blocks

    def _run_tesseract(self, big):
        import pytesseract
        d = pytesseract.image_to_data(big, lang="eng", output_type=pytesseract.Output.DICT)
        words = []
        for i, txt in enumerate(d["text"]):
            if txt.strip() and float(d["conf"][i]) >= self.min_conf:
                x, y, w, h = (d[k][i] for k in ("left", "top", "width", "height"))
                words.append(dict(box=(x, y, x + w, y + h), block=d["block_num"][i],
                                  line=(d["block_num"][i], d["par_num"][i], d["line_num"][i])))
        lines, blocks, keys = [], [], []
        for key in sorted({w["line"] for w in words}):
            ws = [w for w in words if w["line"] == key]
            lines.append(self._union([w["box"] for w in ws]))
            blocks.append(ws[0]["block"])
        return lines, [w["box"] for w in words], blocks

    @staticmethod
    def _union(boxes):
        b = np.array(boxes)
        return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()

    def boxes(self, image):
        """[(level, (x0, y0, x1, y1))] in image coordinates."""
        s = self.ocr_scale
        big = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        lines, words, blocks = (self._run_paddle if self.backend == "paddle" else self._run_tesseract)(big)
        out = []
        if "block" in self.levels:
            for b in sorted({b for b in blocks}):
                out.append(("block", self._union([l for l, bb in zip(lines, blocks) if bb == b])))
        if "line" in self.levels:
            out += [("line", l) for l in lines]
        if "word" in self.levels:
            out += [("word", w) for w in words]
        return [(lvl, tuple(v / s for v in box)) for lvl, box in out]

    def generate(self, image):
        H, W = image.shape[:2]
        masks, seen_boxes, seen_areas = [], set(), set()
        for level, (x0, y0, x1, y1) in self.boxes(image):
            x0, y0 = max(int(x0) - self.pad, 0), max(int(y0) - self.pad, 0)
            x1, y1 = min(int(np.ceil(x1)) + self.pad, W), min(int(np.ceil(y1)) + self.pad, H)
            if (x0, y0, x1, y1) in seen_boxes or x1 <= x0 or y1 <= y0:
                continue  # a block with a single line gives the same box twice: keep one
            seen_boxes.add((x0, y0, x1, y1))
            # DSEG's remove_small_clusters keys masks by pixel area, so equal areas would overwrite each
            # other: widen the box by one column (or row) until its area is unique.
            while (x1 - x0) * (y1 - y0) in seen_areas:
                if x1 < W:
                    x1 += 1
                elif x0 > 0:
                    x0 -= 1
                elif y1 < H:
                    y1 += 1
                else:
                    break
            seg = np.zeros((H, W), bool)
            seg[y0:y1, x0:x1] = True
            seen_areas.add(int(seg.sum()))
            masks.append(dict(segmentation=seg, area=int(seg.sum()), bbox=[x0, y0, x1 - x0, y1 - y0],
                              predicted_iou=1.0, stability_score=1.0, level=level))
        return masks
