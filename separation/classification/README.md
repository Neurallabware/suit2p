# Standalone Classification (Island Extraction)

This folder contains a standalone extraction of the Suite2p soma classification module.

## Public API

- `Classifier`
- `classify`
- `builtin_classfile`
- `user_classfile`

These are exported by `separation/classification/__init__.py`.

## Structure

- `classify.py`, `classifier.py`: copied classification logic.
- `classifiers/classifier.npy`: local builtin classifier artifact.
- `serialization.py`: capture/replay serialization helpers.
- `tools/capture_classification_io.py`: captures real run input/output.
- `tests/test_classification_standalone.py`: exact replay regression test.

## Capture Real I/O

From repo root:

```bash
source /home/yz/anaconda3/etc/profile.d/conda.sh
conda activate suite2p
python -m separation.classification.tools.capture_classification_io
```

This writes:

- `separation/classification/test_input.pkl`
- `separation/classification/test_output.pkl`
- `separation/classification/capture_manifest.json`

## Run Standalone Regression Test

```bash
source /home/yz/anaconda3/etc/profile.d/conda.sh
conda activate suite2p
python -m pytest -c separation/classification/pytest.ini separation/classification/tests/test_classification_standalone.py -s
```

The test prints a classification config summary and `success=True/False`.
