"""Serialization helpers for standalone classification capture/replay."""

from pathlib import Path
from typing import Any

import numpy as np


Primitive = (type(None), bool, int, float, str)


def serialize_object(obj: Any) -> Any:
    """Serialize objects into explicit pickle-safe structures."""
    if isinstance(obj, Primitive):
        return obj

    if isinstance(obj, Path):
        return {"__type__": "path", "value": str(obj)}

    if isinstance(obj, np.generic):
        return {
            "__type__": "numpy_scalar",
            "dtype": str(obj.dtype),
            "value": obj.item(),
        }

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return {
                "__type__": "ndarray_object",
                "shape": list(obj.shape),
                "items": [serialize_object(v) for v in obj.reshape(-1)],
            }
        return {
            "__type__": "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": np.array(obj, copy=True),
        }

    if isinstance(obj, dict):
        return {k: serialize_object(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [serialize_object(v) for v in obj]

    if isinstance(obj, tuple):
        return {"__type__": "tuple", "items": [serialize_object(v) for v in obj]}

    raise TypeError(f"Unsupported object type for serialization: {type(obj)!r}")


def deserialize_object(obj: Any) -> Any:
    """Deserialize captured structures back into runtime objects."""
    if isinstance(obj, Primitive):
        return obj

    if isinstance(obj, list):
        return [deserialize_object(v) for v in obj]

    if isinstance(obj, dict):
        obj_type = obj.get("__type__")
        if obj_type == "path":
            return Path(obj["value"])
        if obj_type == "tuple":
            return tuple(deserialize_object(v) for v in obj["items"])
        if obj_type == "ndarray":
            data = np.array(obj["data"], copy=True)
            data = data.astype(np.dtype(obj["dtype"]), copy=False)
            if tuple(obj["shape"]) != data.shape:
                raise ValueError(
                    f"Shape mismatch during ndarray deserialization: expected {tuple(obj['shape'])}, got {data.shape}"
                )
            return data
        if obj_type == "ndarray_object":
            shape = tuple(obj["shape"])
            items = [deserialize_object(v) for v in obj["items"]]
            data = np.empty(shape, dtype=object)
            data.reshape(-1)[:] = items
            return data
        if obj_type == "numpy_scalar":
            return np.dtype(obj["dtype"]).type(obj["value"])
        return {k: deserialize_object(v) for k, v in obj.items()}

    raise TypeError(f"Unsupported serialized object type: {type(obj)!r}")
