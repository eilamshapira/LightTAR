"""LayerEncoder: pooled hidden states of selected layers of a HuggingFace text encoder."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import numpy as np

LayerSpec = int | float | str
_MODEL_CACHE: dict = {}
_NAMED = {"last": 1.0, "middle": 0.5, "embeddings": 0.0, "first": 0.0}
_PRETRUNCATE = 8  # texts longer than this many characters per token of max_length are cut before tokenizing


def clear_model_cache() -> None:
    """Forget the models loaded by the encoders (frees their memory if nothing else holds them)."""
    _MODEL_CACHE.clear()


def resolve_layers(layers: Sequence[LayerSpec], num_blocks: int) -> list[int]:
    """Map layer specs to absolute indices into ``hidden_states`` (0 = embedding output, num_blocks = last).

    An ``int`` is an index (negative counts from the end), a ``float`` in [0, 1] a relative depth
    ``floor(p * num_blocks + 0.5)``, and a ``str`` one of "last" (1.0), "middle" (0.5), "embeddings" or
    "first" (0.0).
    """
    out = []
    for spec in layers:
        if isinstance(spec, str):
            if spec not in _NAMED:
                raise ValueError(f"unknown layer name {spec!r}; use one of {sorted(_NAMED)}")
            spec = _NAMED[spec]
        if isinstance(spec, (bool, np.bool_)):
            raise TypeError("layer spec cannot be a bool")
        if isinstance(spec, (int, np.integer)):
            idx = int(spec) + (num_blocks + 1 if spec < 0 else 0)
        elif isinstance(spec, (float, np.floating)):
            if not 0.0 <= spec <= 1.0:
                raise ValueError(f"relative layer {spec} must lie in [0, 1]")
            idx = int(math.floor(spec * num_blocks + 0.5))
        else:
            raise TypeError(f"bad layer spec {spec!r}")
        if not 0 <= idx <= num_blocks:
            raise ValueError(f"layer {spec!r} out of range for {num_blocks} blocks")
        out.append(idx)
    return out


def _torch_dtype(dtype: str):
    import torch

    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]


class LayerEncoder:
    """Encode texts with a frozen HuggingFace encoder and return pooled, L2-normalized states of chosen layers.

    Each distinct text is encoded once, in batches of similar token length padded to the longest text;
    the model is loaded on first use and shared by all encoders with the same name, dtype and device.

    Parameters
    ----------
    model_name : str, default "intfloat/e5-small-v2"
        HF model id or local path (anything ``AutoModel`` and ``AutoTokenizer`` load).
    layers : sequence of int, float or str, default (1.0, 0.5)
        Layers to return (see ``resolve_layers``); the default is the last and the middle block. The
        first layer is the projection's anchor layer.
    pooling : {"mean", "cls"}, default "mean"
        Masked mean over tokens, or the first token.
    text_template : str or None, default "passage: {col}: {value}"
        Format string with ``{col}`` and ``{value}``; ``None`` encodes the raw text.
    batch_size : int, default 32
    max_length : int, default 512
        Tokens per text (longer texts are truncated).
    device : str or None, default None
        "cuda" when available, else "cpu".
    dtype : {"fp32", "fp16", "bf16"}, default "fp32"
        Model weights' precision.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        layers: Sequence[LayerSpec] = (1.0, 0.5),
        pooling: str = "mean",
        text_template: str | None = "passage: {col}: {value}",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
        dtype: str = "fp32",
    ):
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")
        if dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError("dtype must be 'fp32', 'fp16' or 'bf16'")
        self.model_name, self.layers, self.pooling = model_name, tuple(layers), pooling
        self.text_template, self.batch_size, self.max_length = text_template, batch_size, max_length
        self.device, self.dtype = device, dtype
        self.model = self.tokenizer = None

    def load(self) -> LayerEncoder:
        """Load the model and tokenizer (idempotent)."""
        import torch

        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        key = (self.model_name, self.dtype, str(self.device_))
        if key not in _MODEL_CACHE:
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModel.from_pretrained(self.model_name).to(device=self.device_, dtype=_torch_dtype(self.dtype))
            _MODEL_CACHE[key] = (mdl.eval(), tok)
        self.model, self.tokenizer = _MODEL_CACHE[key]
        return self

    def _ready(self) -> None:
        if self.model is None or not hasattr(self, "device_"):
            self.load()

    # Frozen weights are never copied: deepcopy and clone share them, pickling drops them.
    def __deepcopy__(self, memo):
        new = copy.copy(self)
        new.layers = tuple(self.layers)
        return new

    def __getstate__(self):
        state = dict(self.__dict__)
        state["model"] = state["tokenizer"] = None
        state.pop("device_", None)
        return state

    @property
    def layer_indices(self) -> list[int]:
        """Absolute ``hidden_states`` indices of ``layers`` (0 = embedding output)."""
        self._ready()
        return resolve_layers(self.layers, int(self.model.config.num_hidden_layers))

    def format(self, texts: Sequence, column: str | None = None) -> list[str]:
        if self.text_template is None:
            return [str(t) for t in texts]
        return [self.text_template.format(col=column or "", value=t) for t in texts]

    def prepare(self, texts: Sequence, column: str | None = None) -> tuple:
        """Format, dedupe and tokenize on the CPU; pass the result to ``encode(..., prepared=...)``."""
        self._ready()
        uniq, inverse = np.unique(np.asarray(self.format(texts, column), dtype=object), return_inverse=True)
        enc = self._tokenize(uniq.tolist())
        ids, masks = enc["input_ids"], enc["attention_mask"]
        order = np.argsort([-len(x) for x in ids], kind="stable")
        return len(uniq), inverse, ids, masks, order

    def encode(self, texts: Sequence, column: str | None = None, output: str = "numpy", prepared: tuple | None = None):
        """Pooled states of ``layers``: an ``(n_texts, n_layers, hidden_size)`` float32 numpy array, or a
        tensor on the encoder's device with ``output="torch"``."""
        import torch

        if output not in ("numpy", "torch"):
            raise ValueError("output must be 'numpy' or 'torch'")
        self._ready()
        idx = self.layer_indices
        n_uniq, inverse, ids, masks, order = prepared if prepared is not None else self.prepare(texts, column)
        dim = int(self.model.config.hidden_size)
        out = torch.empty((n_uniq, len(idx), dim), dtype=torch.float32, device=self.device_)
        order_t = torch.from_numpy(order).to(self.device_)
        pin = str(self.device_).startswith("cuda")
        with torch.inference_mode():
            for s in range(0, n_uniq, self.batch_size):
                rows = order[s : s + self.batch_size]
                ids_np, am_np = self._pad([ids[r] for r in rows], [masks[r] for r in rows])
                ids_t, am = torch.from_numpy(ids_np), torch.from_numpy(am_np)
                if pin:  # asynchronous copy: the host prepares the next batch while the device computes
                    ids_t, am = ids_t.pin_memory(), am.pin_memory()
                ids_t = ids_t.to(self.device_, non_blocking=pin)
                am = am.to(self.device_, non_blocking=pin)
                hs = self.model(input_ids=ids_t, attention_mask=am, output_hidden_states=True).hidden_states
                out[order_t[s : s + self.batch_size]] = self._pool_layers([hs[i] for i in idx], am)
        out = out[torch.from_numpy(np.ascontiguousarray(inverse.ravel())).to(self.device_)]
        return out.cpu().numpy() if output == "numpy" else out

    def _tokenize(self, texts: list) -> dict:
        """Tokenize with truncation. Very long texts are cut at a whitespace first (tokenizing a whole long
        text only to truncate it is slow); a cut text is kept only if it still fills ``max_length`` tokens."""

        def tok(xs):
            return self.tokenizer(xs, truncation=True, max_length=self.max_length, padding=False)

        limit = _PRETRUNCATE * self.max_length
        long = [i for i, t in enumerate(texts) if len(t) > limit]
        if not long:
            return tok(texts)
        cut = list(texts)
        for i in long:
            head = texts[i][:limit]
            j = max(head.rfind(" "), head.rfind("\n"), head.rfind("\t"))
            cut[i] = head[:j] if j > 0 else head
        enc = tok(cut)
        ids, masks = list(enc["input_ids"]), list(enc["attention_mask"])
        redo = [i for i in long if len(ids[i]) < self.max_length]
        if redo:  # the cut text is shorter than max_length tokens: tokenize the full text
            full = tok([texts[i] for i in redo])
            for k, i in enumerate(redo):
                ids[i], masks[i] = full["input_ids"][k], full["attention_mask"][k]
        return {"input_ids": ids, "attention_mask": masks}

    def _pad(self, ids: list, masks: list) -> tuple[np.ndarray, np.ndarray]:
        """Pad a batch to its longest sequence (the tokenizer's pad id and padding side)."""
        width = max(len(x) for x in ids)
        pad_id = self.tokenizer.pad_token_id or 0
        left = getattr(self.tokenizer, "padding_side", "right") == "left"
        out = np.full((len(ids), width), pad_id, dtype=np.int64)
        am = np.zeros((len(ids), width), dtype=np.int64)
        for i, (x, m) in enumerate(zip(ids, masks, strict=True)):
            sl = slice(width - len(x), width) if left else slice(0, len(x))
            out[i, sl], am[i, sl] = x, m
        return out, am

    def _pool_layers(self, hs: list, attention_mask):
        """Pool and L2-normalize each selected layer -> (batch, n_layers, hidden), float32."""
        import torch

        if self.pooling == "cls":
            v = torch.stack([h[:, 0] for h in hs], 1).float()
        else:  # masked mean as one batched matmul per layer, float32 accumulation
            w = attention_mask.to(torch.float32)
            w = (w / w.sum(dim=1, keepdim=True).clamp(min=1e-9)).unsqueeze(1)
            v = torch.cat([torch.bmm(w, h.float()) for h in hs], 1)
        return torch.nn.functional.normalize(v, p=2, dim=2)


class PrecomputedEncoder:
    """Serve encoder states that were computed elsewhere, looked up by the exact cell text.

    Use it as ``LightTAR(encoder=PrecomputedEncoder(vectors))``: ``LightTAR`` hands the encoder each cell
    as a string (``str`` of the DataFrame value, missing values as "Unknown Value"), and those strings are
    the keys. No template is applied. Every lookup in ``transform`` uses the same keys, so the encoder also
    works inside ``Pipeline.predict``. A text with no vector raises ``KeyError``, unless ``fallback`` is
    given; then the missing texts are encoded by it and merged in order.

    Parameters
    ----------
    vectors : mapping
        ``{text: vector}`` shared by all columns, or ``{column: {text: vector}}``; a vector is ``(d,)``
        (one layer) or ``(n_layers, d)``, the same shape for every text of a column.
    fallback : object or None, default None
        Any encoder with ``encode(texts, column)`` returning ``(n, n_layers, d)`` states, for example a
        ``LayerEncoder`` with matching layers.
    """

    def __init__(self, vectors, fallback=None):
        self.vectors = vectors
        self.fallback = fallback

    def __deepcopy__(self, memo):  # the table is read-only and can be large: copies share it
        new = copy.copy(self)
        new.fallback = copy.deepcopy(self.fallback, memo)
        return new

    def _table(self, column) -> Mapping:
        first = next(iter(self.vectors.values()), None)
        if not isinstance(first, Mapping):
            return self.vectors
        if column in self.vectors:
            return self.vectors[column]
        named = {str(k): v for k, v in self.vectors.items()}
        return named.get(str(column), {})

    def encode(self, texts: Sequence, column: str | None = None) -> np.ndarray:
        """Stack the vectors of ``texts`` into an ``(n, n_layers, d)`` float32 array."""
        texts = list(texts)
        table = self._table(column)
        rows: list = [None] * len(texts)
        miss = []
        for i, t in enumerate(texts):
            v = table.get(t)
            if v is None:
                miss.append(i)
            else:
                v = np.asarray(v, dtype=np.float32)
                rows[i] = v[None] if v.ndim == 1 else v
        if miss and self.fallback is None:
            shown = ", ".join(repr(texts[i]) for i in miss[:3])
            raise KeyError(f"{len(miss)} texts of column {column!r} have no precomputed vector, e.g. {shown}")
        if miss:
            filled = np.asarray(self.fallback.encode([texts[i] for i in miss], column), dtype=np.float32)
            for i, v in zip(miss, filled, strict=True):
                rows[i] = v
        if not rows:
            raise ValueError("no texts to encode")
        if len({r.shape for r in rows}) > 1:
            raise ValueError(f"vectors of column {column!r} differ in shape: {sorted({r.shape for r in rows})}")
        return np.stack(rows).astype(np.float32, copy=False)
