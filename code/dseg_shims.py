"""Stub the TensorFlow / efficientnet imports that DSEG-LIME pulls in at module level.

On the SAM path, DSEG_Lime never calls load_img / decode_predictions at runtime; only
`preprocess_input` is touched by ImageExplanation.get_image_and_mask (an EfficientNet
normalisation we do not want on document images, so it becomes the identity).
Call install() BEFORE importing Utilities.dseg_lime.
"""
import importlib.machinery
import sys
import types

from PIL import Image


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)  # find_spec() (used by transformers) needs one
    sys.modules[name] = m
    return m


def _load_img(path, target_size=None, **_):
    img = Image.open(path).convert("RGB")
    return img.resize(target_size[::-1]) if target_size else img


def install():
    tf = _mod("tensorflow")
    keras = _mod("tensorflow.keras")
    utils = _mod("tensorflow.keras.utils", load_img=_load_img)
    apps = _mod("tensorflow.keras.applications")
    eff = _mod("tensorflow.keras.applications.efficientnet", decode_predictions=lambda p, **k: p)
    tf.keras, keras.utils, keras.applications, apps.efficientnet = keras, utils, apps, eff
    en = _mod("efficientnet")
    en.keras = _mod("efficientnet.keras", preprocess_input=lambda x, *a, **k: x)
