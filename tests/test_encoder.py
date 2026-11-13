"""LayerEncoder tests. They need torch and transformers; the e5 tests need intfloat/e5-small-v2 in the
local HF cache and the tiny-model test a tiny BERT (from the cache or the Hub)."""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lighttar import LayerEncoder, LightTAR  # noqa: E402
from lighttar.table import column_texts  # noqa: E402

MODEL = "intfloat/e5-small-v2"
TINY_MODEL = os.environ.get("LIGHTTAR_TEST_HF_MODEL", "hf-internal-testing/tiny-random-BertModel")
TEXTS = [
    "a cheap red bicycle",
    "",
    "nan",
    "The quick brown fox " * 40,
    "a cheap red bicycle",
    "Ünïcødé text — with symbols ✓",
    "short",
    "another moderately long product description " * 5,
] * 9


@pytest.fixture(scope="module")
def enc():
    try:
        return LayerEncoder(MODEL, device="cpu").load()
    except OSError:
        pytest.skip("e5-small-v2 not in the local HF cache")


def reference_forward(texts, model, tok, col, batch_size=32, max_length=512):
    """Straightforward reference: template, pad to max_length, masked mean, L2 norm, every layer."""
    prefixed = [f"passage: {col}: {t}" for t in texts]
    per_layer = None
    for s in range(0, len(prefixed), batch_size):
        b = tok(
            prefixed[s : s + batch_size],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            hs = model(
                input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True
            ).hidden_states
        per_layer = per_layer or [[] for _ in hs]
        mask = b["attention_mask"].unsqueeze(-1).float()
        for j, h in enumerate(hs):
            p = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            per_layer[j].append(torch.nn.functional.normalize(p, p=2, dim=1).numpy())
    return np.stack([np.concatenate(c) for c in per_layer], 1)


def test_matches_reference(enc):
    ref = reference_forward(TEXTS, enc.model, enc.tokenizer, "desc")[:, [-1, -7]]
    out = enc.encode(TEXTS, column="desc")
    assert out.shape == (len(TEXTS), 2, 384) and out.dtype == np.float32
    assert np.abs(out - ref).max() < 1e-5
    np.testing.assert_array_equal(out[0], out[4])  # duplicates are encoded once


def test_template_none_and_cls_pooling(enc):
    out = LayerEncoder(MODEL, device="cpu", text_template=None, pooling="cls", layers=("last",)).encode(["hello world"])
    assert out.shape == (1, 1, 384)
    assert abs(np.linalg.norm(out[0, 0]) - 1.0) < 1e-5  # always L2-normalized
    with pytest.raises(ValueError):
        LayerEncoder(MODEL, pooling="max")


def test_torch_output_equals_numpy(enc):
    t = enc.encode(TEXTS, column="desc", output="torch")
    assert isinstance(t, torch.Tensor)
    np.testing.assert_array_equal(t.cpu().numpy(), enc.encode(TEXTS, column="desc"))
    with pytest.raises(ValueError):
        enc.encode(TEXTS[:2], output="list")


def test_pretruncate_keeps_tokens(enc):
    rng = np.random.default_rng(0)
    words = ["good", "bad", "restaurant", "service", "delicious,", "the", "and", "\n", "Rated", "4.0"]
    texts = [" ".join(rng.choice(words, size=3000)) for _ in range(3)]
    texts += ["x" * 9000, "a " * 3000 + "b" * 6000, "short text"]
    e = LayerEncoder(MODEL, device="cpu", max_length=64).load()
    a = e._tokenize(texts)
    b = e.tokenizer(texts, truncation=True, max_length=64, padding=False)
    assert list(a["input_ids"]) == list(b["input_ids"])
    assert list(a["attention_mask"]) == list(b["attention_mask"])


def test_table_with_real_encoder_and_pickle(enc):
    rng = np.random.default_rng(0)
    words = ["good", "bad", "great", "awful", "fine", "poor"]
    n = 200
    lab = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "price": rng.random(n),
            "review": [" ".join(rng.choice(words[:3] if v else words[3:], 4)) + f" #{i}" for i, v in enumerate(lab)],
            "brand": rng.choice(["a", "b"], n),
        }
    )
    lt = LightTAR(n_anchor=8, n_supervised=2, device="cpu")
    out = lt.fit_transform(df, lab)
    assert lt.text_columns_ == ["review"]
    assert list(out.columns) == ["price"] + [f"review__lt_{i}" for i in range(10)] + ["brand"]
    lt2 = pickle.loads(pickle.dumps(lt))
    assert getattr(lt2, "_encoder", None) is None
    np.testing.assert_allclose(lt2.transform(df).iloc[:, 1:11].to_numpy(), out.iloc[:, 1:11].to_numpy(), atol=1e-5)


def test_several_columns_equal_precomputed_states(enc):
    """Tokenizing the next column in a thread and fitting in another gives the per-column results."""
    rng = np.random.default_rng(1)
    words = ["good", "bad", "great", "awful", "fine", "poor", "food", "staff"]
    df = pd.DataFrame({c: [" ".join(rng.choice(words, size=rng.integers(1, 30))) for _ in range(120)] for c in "abc"})
    y = rng.normal(size=120)
    out = LightTAR(encoder=enc, n_anchor=4, n_supervised=2).fit(df, y).transform(df)
    states = {c: enc.encode(column_texts(df[c]), column=c) for c in "abc"}
    pre = LightTAR(encoder=None, n_anchor=4, n_supervised=2).fit(df, y, layer_states=states)
    np.testing.assert_array_equal(out.to_numpy(), pre.transform(df, layer_states=states).to_numpy())


def test_layer_encoder_tiny_model():
    try:
        enc = LayerEncoder(TINY_MODEL, layers=(1.0, 0.5), device="cpu")
        H = enc.encode(["a short text", "another, longer piece of text", "a short text"], column="desc")
    except OSError as e:  # model not cached and Hub unreachable
        pytest.skip(f"cannot load {TINY_MODEL}: {e}")
    assert H.ndim == 3 and H.shape[:2] == (3, 2)
    assert np.isfinite(H).all()
    assert np.allclose(H[0], H[2])  # identical texts, identical states
