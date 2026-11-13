# LightTAR

LightTAR replaces text or image columns in a pandas DataFrame with numeric features for a tabular model. By default, each selected column contributes up to 30 PCA features from the last encoder layer and 5 supervised features learned from the last and middle layers. The encoder stays frozen.

## Install

Requires Python 3.10 or later. For the text example below, install `lighttar[text]`.

| Use case | Command |
| --- | --- |
| Precomputed embeddings | `pip install lighttar` |
| Text columns | `pip install "lighttar[text]"` |
| Image columns | `pip install "lighttar[image]"` |
| Text, images, and the LightGBM gate | `pip install "lighttar[all]"` |

To use the LightGBM gate with precomputed embeddings, run `pip install "lighttar[gate]"`. You can also combine extras, for example `pip install "lighttar[text,gate]"`.

## Text

The first run downloads the text encoder (`intfloat/e5-small-v2`). Pass the text column names explicitly, or use `LightTAR()` to detect them. Other columns pass through unchanged.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from lighttar import LightTAR

train = pd.DataFrame({"review": ["Very helpful", "Hard to use", "Works well", "Broke immediately"]})
test = pd.DataFrame({"review": ["Easy to use", "Stopped working"]})

model = make_pipeline(LightTAR(columns=["review"]), LogisticRegression())
model.fit(train, [1, 0, 1, 0])
predictions = model.predict(test)
```

## Images

For a column of image filenames, use this transformer in the same pipeline. The default DINOv3 model requires access on Hugging Face.

```python
from lighttar import ImageLayerEncoder, LightTAR

features = LightTAR(columns=["photo"], encoder=ImageLayerEncoder(image_folder="images"))
```

## Precomputed embeddings

Pass one array per split with shape `(rows, layers, embedding_dim)`, with the last encoder layer first. A single layer can also be passed as `(rows, embedding_dim)`.

```python
from lighttar import LightTARProjection

projection = LightTARProjection()
Z_train = projection.fit_transform(H_train, y_train)
Z_test = projection.transform(H_test)
```

## Optional gate

Use `GatedLightTAR` in place of `LightTAR` to choose between LightTAR and PCA using cross-validation. After fitting, `gate_decision_` records the choice.
