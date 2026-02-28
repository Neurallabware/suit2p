# Standalone Registration (Island Extraction)

This folder contains a standalone extraction of the Suite2p registration module.

## Public API

- `registration_wrapper`
- `get_pc_metrics`
- `compute_zpos`
- `highpass_mean_image`

These are exported by `separation/registration/__init__.py`.

## Structure

- `register.py`, `rigid.py`, `nonrigid.py`, `utils.py`, `bidiphase.py`, `metrics.py`, `zalign.py`: copied registration logic.
- `defaults.py`: local defaults provider.
- `logging_utils.py`: local `TqdmToLogger` shim.
- `binary.py`: local `BinaryFile` implementation used by replay.
- `serialization.py`: capture/replay serialization helpers.
- `tools/capture_registration_io.py`: captures real run input/output.
- `tests/test_registration_standalone.py`: exact replay regression test.

## Capture Real I/O

From repo root:

```bash
conda activate suite2p
python -m separation.registration.tools.capture_registration_io
```

This writes:

- `separation/registration/test_input.pkl`
- `separation/registration/test_output.pkl`
- `separation/registration/capture_manifest.json`

## Run Standalone Regression Test

```bash
conda activate suite2p
python -m pytest -c separation/registration/pytest.ini separation/registration/tests/test_registration_standalone.py -s
```

The test prints the registration parameter config and `success=True/False`.
