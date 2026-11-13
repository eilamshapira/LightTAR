"""ImageLayerEncoder tests. The model tests need torch, transformers, Pillow and the DINOv3 ViT-S/16 weights
in the local HF cache, and are skipped otherwise. They run on the CPU."""

import numpy as np
import pandas as pd
import pytest

from lighttar import detect_image_columns

MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def test_detect_image_columns():
    df = pd.DataFrame(
        {
            "img": ["a/1.jpg", "b/2.PNG", None, "x.webp?w=720"],
            "txt": ["a.jpg", "hello", "b.png", "c.jpg"],
            "num": [1, 2, 3, 4],
        }
    )
    assert detect_image_columns(df) == ["img"]


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
Image = pytest.importorskip("PIL.Image")

from lighttar import ImageLayerEncoder, LightTAR  # noqa: E402


@pytest.fixture(scope="module")
def enc():
    try:
        return ImageLayerEncoder(MODEL, layers=(-1, -7), device="cpu").load()
    except (OSError, ValueError):
        pytest.skip("DINOv3 ViT-S/16 not in the local HF cache")


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("imgs")
    rng = np.random.default_rng(0)
    names = []
    for i in range(12):
        h, w = (int(rng.integers(40, 700)), int(rng.integers(40, 700)))
        im = Image.fromarray((rng.random((h, w, 3)) * 255).astype(np.uint8))
        if i % 4 == 1:
            im = im.convert("L")
        elif i % 4 == 3:  # palette images only as png (JPEG cannot store mode P)
            im = im.convert("P")
        name = f"im{i}.png" if i % 2 else f"im{i}.jpg"
        im.save(d / name)
        names.append(name)
    return d, names


def reference_states(paths, model, processor):
    """RGB, thumbnail to 512 with LANCZOS, the image processor, CLS of every hidden state."""
    imgs = []
    for p in paths:
        img = Image.open(p)
        img = (img.convert("RGBA") if img.mode == "P" else img).convert("RGB")
        if max(img.size) > 512:
            img.thumbnail((512, 512), Image.LANCZOS)
        imgs.append(img)
    with torch.no_grad():
        hs = model(**processor(images=imgs, return_tensors="pt"), output_hidden_states=True).hidden_states
    return np.stack([h[:, 0].numpy() for h in hs], 1).astype(np.float32)


def test_matches_reference(enc, image_dir):
    d, names = image_dir
    ref = reference_states([d / n for n in names], enc.model, enc.processor)[:, [-1, -7]]
    out = ImageLayerEncoder(MODEL, layers=(1.0, 0.5), image_folder=str(d), device="cpu", batch_size=5).encode(names)
    assert out.shape == (len(names), 2, 384) and out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_dedupe_and_torch_output(enc, image_dir):
    d, names = image_dir
    rows = names + names[:5]
    e = ImageLayerEncoder(MODEL, image_folder=str(d), device="cpu", batch_size=4)
    out = e.encode(rows, output="torch")
    assert isinstance(out, torch.Tensor) and tuple(out.shape) == (len(rows), 2, 384)
    np.testing.assert_array_equal(out[len(names) :].numpy(), out[:5].numpy())
    np.testing.assert_array_equal(out.numpy(), e.encode(rows))


def test_pil_input_and_bytes_paths(enc, image_dir):
    d, names = image_dir
    pil = [Image.open(d / n) for n in names[:4]]
    a = ImageLayerEncoder(MODEL, device="cpu").encode(pil)
    b = ImageLayerEncoder(MODEL, image_folder=str(d), device="cpu").encode([n.encode() for n in names[:4]])
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_missing_value_raises(enc):
    with pytest.raises(ValueError, match="missing image"):
        ImageLayerEncoder(MODEL, device="cpu").encode(["a.jpg", None])


def test_table_with_image_column(enc, image_dir):
    d, names = image_dir
    rows = (names * 5)[:60]
    df = pd.DataFrame({"pic": rows, "num": np.arange(60.0)})
    y = np.arange(60) % 2
    e = ImageLayerEncoder(MODEL, image_folder=str(d), device="cpu")
    t = LightTAR(columns=["pic"], encoder=e, n_anchor=4, n_supervised=2)
    out = t.fit_transform(df, y)
    assert list(out.columns) == [f"pic__lt_{i}" for i in range(6)] + ["num"]
    assert np.isfinite(t.transform(df.iloc[:10]).filter(like="pic__").to_numpy()).all()
