"""Regression tests for standalone registration replay."""

from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
import pickle
import re
from typing import Any, Dict, Iterable, List

import numpy as np

from separation.registration.binary import BinaryFile
from separation.registration.serialization import deserialize_object


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+suite2p\b")


class BlockSuite2PImports:
    """Context manager that blocks suite2p imports during standalone replay."""

    def __enter__(self):
        self._orig_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "suite2p" or name.startswith("suite2p."):
                raise ImportError(f"Blocked import by island rule: {name}")
            return self._orig_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.__import__ = self._orig_import



def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" not in path.parts:
            yield path



def _assert_no_suite2p_import_statements(root: Path) -> None:
    offenders: List[str] = []
    for py_path in _iter_python_files(root):
        for lineno, line in enumerate(py_path.read_text().splitlines(), start=1):
            if IMPORT_PATTERN.search(line):
                offenders.append(f"{py_path}:{lineno}: {line.strip()}")
    if offenders:
        detail = "\n".join(offenders)
        raise AssertionError(f"Island rule violation: found suite2p imports\n{detail}")



def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open("rb") as f:
        return pickle.load(f)



def _build_binary_file(desc: Dict[str, Any], opened: List[BinaryFile]) -> BinaryFile:
    for key in ("filename", "Ly", "Lx", "n_frames", "dtype", "write"):
        if key not in desc:
            raise ValueError(f"Binary descriptor missing key '{key}': {desc}")
    filename = Path(desc["filename"])
    if not filename.exists():
        raise FileNotFoundError(f"Binary file not found for descriptor: {filename}")
    bf = BinaryFile(
        Ly=int(desc["Ly"]),
        Lx=int(desc["Lx"]),
        filename=str(filename),
        n_frames=int(desc["n_frames"]),
        dtype=str(desc["dtype"]),
        write=bool(desc["write"]),
    )
    opened.append(bf)
    return bf



def _compare_exact(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            raise AssertionError(f"{path}: expected ndarray, got {type(actual)}")
        if expected.dtype != actual.dtype:
            raise AssertionError(f"{path}: dtype mismatch {expected.dtype} != {actual.dtype}")
        if expected.shape != actual.shape:
            raise AssertionError(f"{path}: shape mismatch {expected.shape} != {actual.shape}")
        if not np.array_equal(expected, actual):
            raise AssertionError(f"{path}: ndarray values differ")
        return

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected dict, got {type(actual)}")
        if set(expected.keys()) != set(actual.keys()):
            raise AssertionError(
                f"{path}: key mismatch {set(expected.keys())} != {set(actual.keys())}"
            )
        for key in expected:
            _compare_exact(expected[key], actual[key], path=f"{path}.{key}")
        return

    if isinstance(expected, tuple):
        if not isinstance(actual, tuple):
            raise AssertionError(f"{path}: expected tuple, got {type(actual)}")
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: tuple length mismatch {len(expected)} != {len(actual)}")
        for i, (exp_v, act_v) in enumerate(zip(expected, actual)):
            _compare_exact(exp_v, act_v, path=f"{path}[{i}]")
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise AssertionError(f"{path}: expected list, got {type(actual)}")
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: list length mismatch {len(expected)} != {len(actual)}")
        for i, (exp_v, act_v) in enumerate(zip(expected, actual)):
            _compare_exact(exp_v, act_v, path=f"{path}[{i}]")
        return

    if type(expected) is not type(actual):
        raise AssertionError(f"{path}: type mismatch {type(expected)} != {type(actual)}")
    if expected != actual:
        raise AssertionError(f"{path}: value mismatch {expected!r} != {actual!r}")



def test_registration_standalone_exact_replay():
    _assert_no_suite2p_import_statements(PACKAGE_ROOT)

    captured_input = _load_pickle(PACKAGE_ROOT / "test_input.pkl")
    captured_output = _load_pickle(PACKAGE_ROOT / "test_output.pkl")

    opened: List[BinaryFile] = []
    kwargs = deserialize_object(
        captured_input,
        binary_file_factory=lambda d: _build_binary_file(d, opened),
    )
    expected_output = deserialize_object(captured_output)

    registration_settings = kwargs.get("settings", {})
    print("registration_settings=")
    print(json.dumps(registration_settings, indent=2, sort_keys=True, default=str))

    success = False
    try:
        with BlockSuite2PImports():
            standalone_module = importlib.import_module("separation.registration")
            actual_output = standalone_module.registration_wrapper(**kwargs)
        _compare_exact(expected_output, actual_output)
        success = True
    finally:
        for bf in opened:
            bf.close()

    print(f"success={success}")
    assert success
