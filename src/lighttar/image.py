"""ImageLayerEncoder: pooled hidden states of selected layers of a HuggingFace vision encoder."""

from __future__ import annotations

import copy
import io
import os
from collections.abc import Sequence

import numpy as np

from .encoder import _MODEL_CACHE, LayerSpec, _torch_dtype, resolve_layers

DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".jpe")
_MAX_IMAGE_SIZE = 512  # longest side after loading (thumbnail with LANCZOS)


def detect_image_columns(df) -> list:
    """Columns whose non-empty, non-missing values are all strings containing an image file suffix
    (``.jpg``, ``.png``, ...; a substring match, so URLs with query strings count)."""
    cols = []
    for col in df.columns:
        s = df[col].dropna()
        s = s[s != ""]
        if all(isinstance(v, str) and any(x in v.lower() for x in IMAGE_SUFFIXES + (".gz",)) for v in s):
            cols.append(col)
    return cols


def _is_missing(v) -> bool:
    if v is None:
        return True
    import pandas as pd

    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


class ImageLayerEncoder:
    """Encode images with a frozen HuggingFace vision encoder and return the CLS token of chosen layers.

    Images are paths (relative to ``image_folder`` if given), byte strings holding a utf-8 path, or PIL
    images. Each is loaded as RGB, shrunk to at most 512 pixels on its longest side, passed through the
    model's ``AutoImageProcessor``, and its CLS token is read from each selected hidden state (no
    normalization). Each distinct image is encoded once; files are read and decoded in threads while the
    device runs the previous batch.

    Parameters
    ----------
    model_name : str, default "facebook/dinov3-vits16-pretrain-lvd1689m"
        HF model id or local path (``AutoModel`` and ``AutoImageProcessor``).
    layers : sequence of int, float or str, default (1.0, 0.5)
        As in ``LayerEncoder``; the first layer is the projection's anchor layer.
    image_folder : str or None, default None
        Folder that relative paths are joined to.
    batch_size : int, default 64
    device : str or None, default None
        "cuda" when available, else "cpu".
    dtype : {"fp32", "fp16", "bf16"}, default "fp32"
        Model weights' precision.
    """

    def __init__(
        self,
        model_name: str = DINOV3_SMALL,
        layers: Sequence[LayerSpec] = (1.0, 0.5),
        image_folder: str | None = None,
        batch_size: int = 64,
        device: str | None = None,
        dtype: str = "fp32",
    ):
        if dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError("dtype must be 'fp32', 'fp16' or 'bf16'")
        self.model_name, self.layers, self.image_folder = model_name, tuple(layers), image_folder
        self.batch_size, self.device, self.dtype = batch_size, device, dtype
        self.model = self.processor = None

    def load(self) -> ImageLayerEncoder:
        """Load the model and image processor (idempotent)."""
        import torch

        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        key = ("image", self.model_name, self.dtype, str(self.device_))
        if key not in _MODEL_CACHE:
            from transformers import AutoImageProcessor, AutoModel

            proc = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
            mdl = AutoModel.from_pretrained(self.model_name).to(device=self.device_, dtype=_torch_dtype(self.dtype))
            _MODEL_CACHE[key] = (mdl.eval(), proc)
        self.model, self.processor = _MODEL_CACHE[key]
        return self

    def _ready(self) -> None:
        if self.model is None or not hasattr(self, "device_"):
            self.load()

    def __deepcopy__(self, memo):
        new = copy.copy(self)
        new.layers = tuple(self.layers)
        return new

    def __getstate__(self):
        state = dict(self.__dict__)
        state["model"] = state["processor"] = None
        state.pop("device_", None)
        return state

    @property
    def layer_indices(self) -> list[int]:
        """Absolute ``hidden_states`` indices of ``layers`` (0 = patch-embedding output)."""
        self._ready()
        return resolve_layers(self.layers, int(self.model.config.num_hidden_layers))

    def _key(self, v):
        """Dedupe key and loadable item of one input value."""
        if _is_missing(v):
            raise ValueError("missing image value (None/NaN); fill or drop it before encoding")
        if isinstance(v, bytes):  # a utf-8 path
            v = v.decode("utf-8")
        if isinstance(v, os.PathLike):
            v = os.fspath(v)
        if isinstance(v, str):
            path = os.path.join(self.image_folder, v) if self.image_folder is not None else v
            return ("path", path), path
        return ("obj", id(v)), v  # a PIL image

    @staticmethod
    def _load_one(item):
        from PIL import Image

        if isinstance(item, str):
            with open(item, "rb") as f:
                item = Image.open(io.BytesIO(f.read()))
        img = item.convert("RGBA") if item.mode == "P" else item
        img = img.convert("RGB")  # always a new image: the caller's object is never modified
        if max(img.size) > _MAX_IMAGE_SIZE:
            img.thumbnail((_MAX_IMAGE_SIZE, _MAX_IMAGE_SIZE), Image.LANCZOS)
        return img

    def _load_batch(self, items, pool):
        import warnings

        imgs = list(pool.map(self._load_one, items))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.processor(images=imgs, return_tensors="pt")["pixel_values"]

    def encode(self, images: Sequence, column: str | None = None, output: str = "numpy"):
        """Pooled states of ``layers``: an ``(n_images, n_layers, hidden_size)`` float32 numpy array, or a
        tensor on the encoder's device with ``output="torch"``. ``column`` is unused."""
        from concurrent.futures import ThreadPoolExecutor

        import torch

        if output not in ("numpy", "torch"):
            raise ValueError("output must be 'numpy' or 'torch'")
        self._ready()
        idx = self.layer_indices
        tdt = next(self.model.parameters()).dtype
        items, inverse, seen = [], np.empty(len(images), dtype=np.int64), {}
        for i, v in enumerate(images):
            k, item = self._key(v)
            j = seen.get(k)
            if j is None:
                j = seen[k] = len(items)
                items.append(item)
            inverse[i] = j
        n_uniq = len(items)
        dim = int(self.model.config.hidden_size)
        out = torch.empty((n_uniq, len(idx), dim), dtype=torch.float32, device=self.device_)
        batches = [items[s : s + self.batch_size] for s in range(0, n_uniq, self.batch_size)]
        pin = str(self.device_).startswith("cuda")
        with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool, ThreadPoolExecutor(1) as ahead:
            nxt = ahead.submit(self._load_batch, batches[0], pool) if batches else None
            with torch.inference_mode():
                for b in range(len(batches)):
                    pv = nxt.result()
                    if b + 1 < len(batches):  # load the next batch while the device runs this one
                        nxt = ahead.submit(self._load_batch, batches[b + 1], pool)
                    if pin:
                        pv = pv.pin_memory()
                    pv = pv.to(self.device_, dtype=tdt, non_blocking=pin)
                    hs = self.model(pixel_values=pv, output_hidden_states=True).hidden_states
                    s = b * self.batch_size
                    out[s : s + len(batches[b])] = torch.stack([hs[i][:, 0] for i in idx], 1).float()
        if n_uniq != len(images):
            out = out[torch.from_numpy(inverse).to(self.device_)]
        return out.cpu().numpy() if output == "numpy" else out
