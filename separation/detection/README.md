# Standalone Detection (Island Extraction)

This folder contains a standalone extraction of the Suite2p detection module.

## Public API

- `detection_wrapper`
- `bin_movie`
- `roi_stats`
- `assign_overlaps`

These are exported by `separation/detection/__init__.py`.

## Structure

- `detect.py`, `sparsedetect.py`, `sourcery.py`, `stats.py`, `utils.py`, `chan2detect.py`, `denoise.py`, `anatomical.py`, `metrics.py`: copied detection logic.
- `classification/`: local classifier implementation copy used by detection preclassify.
- `extraction/masks.py`: local mask helpers for chan2 detection.
- `classifiers/classifier.npy`: local builtin classifier data file.
- `nonrigid_compat.py`: local registration helper subset needed by detection denoise.
- `defaults.py`: local defaults provider.
- `logging_utils.py`: local `TqdmToLogger` shim.
- `binary.py`: local `BinaryFile` implementation used by replay.
- `serialization.py`: capture/replay serialization helpers.
- `tools/capture_detection_io.py`: captures real run input/output.
- `tests/test_detection_standalone.py`: exact replay + registration->detection compatibility tests.

## Capture Real I/O

From repo root:

```bash
source /home/yz/anaconda3/etc/profile.d/conda.sh
conda activate suite2p
python -m separation.detection.tools.capture_detection_io
```

This writes:

- `separation/detection/test_input.pkl`
- `separation/detection/test_output.pkl`
- `separation/detection/capture_manifest.json`

## Run Standalone Regression Tests

```bash
source /home/yz/anaconda3/etc/profile.d/conda.sh
conda activate suite2p
python -m pytest -c separation/detection/pytest.ini separation/detection/tests/test_detection_standalone.py -s
```

The tests print detection parameter config and success flags.
