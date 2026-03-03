# -*- coding: utf-8 -*-
"""
Model I/O
---------
Secure model serialization with .skops (preferred) or .joblib/.pkl fallback.

Why .skops over .joblib / .pkl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
joblib and pickle serialize Python objects by encoding the *bytecode* needed
to reconstruct them.  That means loading a .joblib or .pkl file can execute
arbitrary Python code — a malicious model file received from a colleague or
downloaded from the internet is indistinguishable from a valid one until it is
already running on your machine.

.skops (SKlearn Object Persistence System) takes a different approach:
  - It stores the model in a JSON-like schema that describes the object graph
    without embedding executable bytecode.
  - Before loading, you call ``get_untrusted_types()`` to get the full list of
    Python types referenced in the file.  You then decide explicitly which of
    those types you ``trust``.  If you did not put them there, you can refuse.
  - This makes supply-chain attacks on model files impossible: even a crafted
    .skops file cannot run code until you explicitly trust a dangerous type.
  - .skops is also the recommended format for sharing sklearn models on
    Hugging Face Hub.

Rule of thumb:
  • Use .skops  for any model you share, deploy to production, or keep in git.
  • Use .joblib as a fast local cache that you control end-to-end.
  • Avoid .pkl  except for temporary scratch files.
"""

import joblib
from pathlib import Path
from typing import Any, List, Union

try:
    import skops.io as sio
    _SKOPS_AVAILABLE = True
except ImportError:  # skops is optional
    sio = None  # type: ignore[assignment]
    _SKOPS_AVAILABLE = False

SUPPORTED_EXTENSIONS = {".skops", ".joblib", ".pkl"}


def save_model_file(obj: Any, path: Union[str, Path]) -> Path:
    """
    Save a model object to disk.

    The serialization format is determined by the file extension:

    =========  =====================================================
    .skops     Secure JSON-based format — recommended for sharing /
               deployment.  Requires ``skops`` package.
    .joblib    Fast binary pickle — local cache only.
    .pkl       Standard pickle — least preferred.
    =========  =====================================================

    Args:
        obj:  Any sklearn-compatible object or plain dict.
        path: Destination path.  The extension chooses the format.

    Returns:
        Resolved ``Path`` that was written.

    Raises:
        ImportError: If .skops is requested but ``skops`` is not installed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".skops":
        if not _SKOPS_AVAILABLE:
            raise ImportError(
                "The 'skops' package is not installed.\n"
                "Install it with:  uv add skops"
            )
        assert sio is not None
        sio.dump(obj, path)
    else:
        # .joblib or .pkl — both use joblib (faster compression than raw pickle)
        joblib.dump(obj, path)

    return path.resolve()


def load_model_file(path: Union[str, Path]) -> Any:
    """
    Load a model object from disk.

    For .skops files all types present in the file are auto-trusted, which is
    appropriate for models *you* saved yourself.  If you received a .skops
    file from an untrusted source, call ``audit_skops_file()`` first to
    inspect the embedded types before deciding whether to load.

    Args:
        path: Path to the model file (.skops, .joblib, or .pkl).

    Returns:
        Deserialized object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError:       If .skops is requested but ``skops`` is not installed.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix == ".skops":
        if not _SKOPS_AVAILABLE:
            raise ImportError(
                "The 'skops' package is not installed.\n"
                "Install it with:  uv add skops"
            )
        assert sio is not None
        # Auto-trust all types found in our own files.
        # For third-party files use audit_skops_file() before this step.
        untrusted_types = sio.get_untrusted_types(file=str(path))
        return sio.load(str(path), trusted=untrusted_types)
    else:
        return joblib.load(path)


def audit_skops_file(path: Union[str, Path]) -> List[str]:
    """
    Return the list of Python types embedded in a .skops file.

    Use this to inspect a third-party model file *before* loading it.
    If the list contains unexpected types, do not load the file.

    Example::

        types = audit_skops_file("received_model.skops")
        # Inspect types — e.g. ['sklearn.ensemble._iforest.IsolationForest', ...]
        trusted = [t for t in types if t.startswith("sklearn.")]
        obj = skops.io.load("received_model.skops", trusted=trusted)

    Args:
        path: Path to a .skops file.

    Returns:
        List of fully-qualified Python type names referenced in the file.
    """
    if not _SKOPS_AVAILABLE:
        raise ImportError(
            "The 'skops' package is not installed.\n"
            "Install it with:  uv add skops"
        )
    assert sio is not None
    return sio.get_untrusted_types(file=str(path))


def skops_available() -> bool:
    """Return True if the ``skops`` package is importable."""
    return _SKOPS_AVAILABLE
