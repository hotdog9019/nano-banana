"""
Local Python startup customization.

This repository sometimes runs under a Windows sandbox where directories created
with `os.mkdir(..., 0o700)` (used by `tempfile.mkdtemp`) end up not being
writable by the current process. This breaks `pip` and other tooling that
relies on `tempfile.TemporaryDirectory`.

Opt-in by setting `NANO_BANANA_TEMPFILE_FIX=1` and adding this folder to
`PYTHONPATH` (see `tools/install_ml_deps.ps1`).
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Optional


def _patch_tempfile_mkdtemp() -> None:
    original_mkdtemp = tempfile.mkdtemp

    def mkdtemp(
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[str] = None,
    ) -> str:
        prefix2, suffix2, dir2, output_type = tempfile._sanitize_params(prefix, suffix, dir)  # type: ignore[attr-defined]

        names = tempfile._get_candidate_names()  # type: ignore[attr-defined]
        if output_type is bytes:
            names = map(os.fsencode, names)

        for _ in range(tempfile.TMP_MAX):
            name = next(names)
            path = os.path.join(dir2, prefix2 + name + suffix2)
            sys.audit("tempfile.mkdtemp", path)
            try:
                # Avoid passing 0o700 on Windows (see module docstring).
                os.mkdir(path)
            except FileExistsError:
                continue
            except PermissionError:
                # On Windows, this can mean "already exists" too.
                if os.path.isdir(path):
                    continue
                raise
            return path

        raise FileExistsError(f"No usable temporary directory name found in {tempfile.TMP_MAX} attempts")

    tempfile.mkdtemp = mkdtemp  # type: ignore[assignment]
    # Keep a reference to the original for debugging.
    tempfile._nano_banana_original_mkdtemp = original_mkdtemp  # type: ignore[attr-defined]


def _should_patch() -> bool:
    if os.name != "nt":
        return False
    if os.environ.get("NANO_BANANA_TEMPFILE_FIX", "").strip() not in {"1", "true", "True", "yes", "YES"}:
        return False
    return True


if _should_patch():
    _patch_tempfile_mkdtemp()
