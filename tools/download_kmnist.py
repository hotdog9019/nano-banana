from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import random
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


KMNIST_BASE_URLS = [
    # torchvision uses http; try https first in case the environment prefers it.
    "https://codh.rois.ac.jp/kmnist/dataset/kmnist/",
    "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
]

# Filenames + MD5 from torchvision.datasets.KMNIST resources.
KMNIST_RESOURCES: list[tuple[str, str]] = [
    ("train-images-idx3-ubyte.gz", "bdb82020997e1d708af4cf47b453dcf7"),
    ("train-labels-idx1-ubyte.gz", "e144d726b3acfaa3e44228e80efcd344"),
    ("t10k-images-idx3-ubyte.gz", "5c965bf0a639b31b8f53240b1b52f4d7"),
    ("t10k-labels-idx1-ubyte.gz", "7320c461ea6c1c855c0b718fb2a4b134"),
]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_resume(url: str, dest: Path, *, md5: str, timeout_s: float, retries: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and _md5(dest) == md5:
        return

    user_agent = "nano-banana-kmnist-downloader/1.0 (+python urllib)"
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            resume_from = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": user_agent}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
                mode = "ab" if resume_from > 0 else "wb"
                with part.open(mode) as f:
                    shutil.copyfileobj(resp, f)

            if _md5(part) != md5:
                raise ValueError(f"MD5 mismatch for {dest.name}")

            part.replace(dest)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionResetError, TimeoutError, OSError) as e:
            last_err = e
            # Small jittered backoff to avoid hammering the host.
            sleep_s = min(30.0, 0.75 * (2 ** (attempt - 1))) + random.random() * 0.25
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download {dest.name} after {retries} retries") from last_err


def _gunzip(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download KMNIST in the layout expected by torchvision.datasets.KMNIST")
    ap.add_argument("--root", default="data", help="Dataset root dir (will create KMNIST/raw under it)")
    ap.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds)")
    ap.add_argument("--retries", type=int, default=10, help="Retries per file")
    ap.add_argument("--no-extract", action="store_true", help="Only download .gz files, do not gunzip")
    args = ap.parse_args()

    root = Path(args.root)
    raw_dir = root / "KMNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, md5 in KMNIST_RESOURCES:
        dest = raw_dir / filename
        ok = False
        last_err: Exception | None = None
        for base in KMNIST_BASE_URLS:
            url = base + filename
            try:
                print(f"Downloading {url} -> {dest}")
                _download_with_resume(url, dest, md5=md5, timeout_s=args.timeout, retries=args.retries)
                ok = True
                break
            except Exception as e:
                last_err = e
                continue
        if not ok:
            raise RuntimeError(f"All mirrors failed for {filename}") from last_err

        if not args.no_extract:
            out = raw_dir / filename.removesuffix(".gz")
            print(f"Extracting {dest.name} -> {out.name}")
            _gunzip(dest, out)

    print(f"Done. torchvision should now work with root={root} and download=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

