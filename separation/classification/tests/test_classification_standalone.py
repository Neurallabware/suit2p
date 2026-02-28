"""Regression tests for standalone classification replay."""

from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
import pickle
import re
from typing import Any, Iterable, List

import numpy as np

from separation.classification.serialization import deserialize_object


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+suite2p\b")
LOCAL_BUILTIN_CLASSFILE = PACKAGE_ROOT / "classifiers" / "classifier.npy"


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


def _as_path(value: Any) -> Path | None:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    return None


def _normalize_classfile(kwargs: dict) -> dict:
    classfile = _as_path(kwargs.get("classfile"))
    if classfile is None:
        return kwargs

    classfile_posix = classfile.as_posix()
    if classfile_posix.endswith("/suite2p/classifiers/classifier.npy"):
        kwargs["classfile"] = LOCAL_BUILTIN_CLASSFILE
    return kwargs


def _compare_exact(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            raise AssertionError(f"{path}: expected ndarray, got {type(actual)}")
        if expected.dtype != actual.dtype:
            raise AssertionError(f"{path}: dtype mismatch {expected.dtype} != {actual.dtype}")
        if expected.shape != actual.shape:
            raise AssertionError(f"{path}: shape mismatch {expected.shape} != {actual.shape}")
        if expected.dtype == object:
            for idx in np.ndindex(expected.shape):
                _compare_exact(expected[idx], actual[idx], path=f"{path}[{idx}]")
        elif not np.array_equal(expected, actual, equal_nan=True):
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


def test_classification_standalone_exact_replay():
    _assert_no_suite2p_import_statements(PACKAGE_ROOT)

    captured_input = _load_pickle(PACKAGE_ROOT / "test_input.pkl")
    captured_output = _load_pickle(PACKAGE_ROOT / "test_output.pkl")

    kwargs = deserialize_object(captured_input)
    expected_output = deserialize_object(captured_output)

    if not isinstance(kwargs, dict):
        raise TypeError(f"Expected deserialized kwargs to be dict, got {type(kwargs)}")
    kwargs = _normalize_classfile(kwargs)

    summary = {
        "classfile": str(kwargs.get("classfile")),
        "keys": kwargs.get("keys"),
        "n_stat": len(kwargs.get("stat", [])),
    }
    print("classification_config=")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))

    success = False
    with BlockSuite2PImports():
        standalone_module = importlib.import_module("separation.classification")
        actual_output = standalone_module.classify(**kwargs)
    _compare_exact(expected_output, actual_output)
    success = True

    print(f"success={success}")
    assert success
