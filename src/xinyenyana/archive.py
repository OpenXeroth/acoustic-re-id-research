"""Putting a result where losing one disk does not lose the evidence.

Every number in [`docs/measured.md`](../../docs/measured.md) comes from a result
file. Until this module existed those files lived only on xen1's own disk: not
in Git, not in the bucket. A disk failure would have turned every published
figure into an assertion, recoverable only by re-running work that needs a GPU
and the benchmark archives.

Two things are fixed here.

**Where the file lives.** A result is copied into the archive under the digest
of its own bytes. Identical bytes land on the same object, so a repeat is a
no-op; different bytes are a different object, so nothing is ever overwritten.

**Which code produced it.** A result records the commit it ran at, and that
commit stops existing when the branch is squash-merged, which is how this
project merges. Every result file written on a branch therefore names a
revision the repository no longer contains. The source digest below is taken
over the measurement code itself and survives any merge strategy.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

#: Where archived results go. Overridden by ``XINYENYANA_RESULT_ARCHIVE``, and
#: unset means no archiving, so tests and local runs never reach the network.
DEFAULT_PREFIX = "gs://xinyenyana/results"
ARCHIVE_VARIABLE = "XINYENYANA_RESULT_ARCHIVE"

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=False, capture_output=True, text=True)


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-serialisable value with separators and key order fixed.

    Held here once. Three modules carried their own copy, one of them with a
    docstring saying it was identical to another, which is what a duplicated
    definition looks like from the inside.
    """

    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def sha256_file(path: Path, *, block: int = 1 << 20) -> str:
    """The SHA-256 of a file, read in blocks so its size does not matter.

    ``block`` is settable so a test can exercise the loop on a small file.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(root: Path) -> str:
    """A digest of the measurement code, independent of how it was merged.

    Taken over every Python file under the package, by relative path and
    content, so a result carries an identifier for the code that produced it
    that no rebase, squash or branch deletion can invalidate.
    """

    files = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"no source files under {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def archive_object_name(path: Path, *, prefix: str) -> str:
    """Where a result file belongs in the archive: under the digest of its bytes."""

    return f"{prefix.rstrip('/')}/{sha256_file(path)}.json"


def _token_copy_and_describe(
    path: Path, target: str, token_file: Path, *, transport: httpx.BaseTransport | None
) -> tuple[Any, str]:
    """Use exact-object JSON API requests; bucket listing permission is unnecessary."""

    location = urlsplit(target)
    if location.scheme != "gs" or not location.netloc or location.query or location.fragment:
        raise ValueError("the archive must be an unversioned gs:// object prefix")
    bucket, name = quote(location.netloc, safe=""), location.path.lstrip("/")
    base = "https://storage.googleapis.com"
    try:
        token = token_file.read_text().strip()
        if not token:
            raise ValueError("archive access token is empty")
        headers = {"Authorization": f"Bearer {token}"}
        with httpx.Client(transport=transport, timeout=120, follow_redirects=False) as client:
            with path.open("rb") as handle:
                copied = client.post(
                    f"{base}/upload/storage/v1/b/{bucket}/o",
                    params={"uploadType": "media", "name": name, "ifGenerationMatch": "0"},
                    headers={
                        **headers,
                        "Content-Type": "application/json",
                        "Content-Length": str(path.stat().st_size),
                    },
                    content=iter(lambda: handle.read(1 << 20), b""),
                )
            described = client.get(
                f"{base}/storage/v1/b/{bucket}/o/{quote(name, safe='')}", headers=headers
            )
            if described.status_code != 200:
                return None, (
                    f"archive upload HTTP {copied.status_code}; "
                    f"metadata verification HTTP {described.status_code}"
                )
            return described.json(), ""
    except (OSError, ValueError, httpx.HTTPError) as error:
        # Do not include request/response dumps or authentication headers.
        return None, f"archive request failed: {type(error).__name__}"


def archive_result(
    path: Path,
    *,
    prefix: str,
    run: Runner = _run,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Create a content-addressed object and verify its size and transfer checksum.

    With ``CLOUDSDK_AUTH_ACCESS_TOKEN_FILE``, use the JSON API directly so the
    experiment needs only create/get permissions on its results prefix.
    Otherwise use the operator's gcloud configuration. Existing objects are
    accepted only when their metadata matches these bytes.
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    target = archive_object_name(path, prefix=prefix)
    token_file = os.environ.get("CLOUDSDK_AUTH_ACCESS_TOKEN_FILE")
    if token_file and run is _run:
        metadata, error = _token_copy_and_describe(
            path, target, Path(token_file), transport=transport
        )
        if metadata is None:
            return {"archived": False, "object": target, "error": error}
    else:
        result = run(
            ["gcloud", "storage", "cp", "--quiet", "--if-generation-match=0", str(path), target]
        )
        # A repeat can return 412 because generation zero refuses an existing
        # object. Its verified bytes establish whether the result is safe.
        listed = run(["gcloud", "storage", "objects", "describe", target, "--format=json"])
        if listed.returncode != 0:
            return {
                "archived": False,
                "object": target,
                "error": (
                    (result.stderr or result.stdout)
                    if result.returncode
                    else (listed.stderr or listed.stdout)
                ).strip()[:500],
            }
        try:
            metadata = json.loads(listed.stdout)
        except ValueError:
            return {"archived": False, "object": target, "error": "invalid archive object metadata"}
    try:
        remote_size = int(metadata["size"])
        remote_md5 = str(metadata.get("md5_hash", metadata.get("md5Hash", "")))
        generation = str(metadata["generation"])
    except (ValueError, KeyError, TypeError):
        return {"archived": False, "object": target, "error": "invalid archive object metadata"}
    checksum = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            checksum.update(chunk)
    local_md5 = base64.b64encode(checksum.digest()).decode()
    if remote_size != path.stat().st_size or remote_md5 != local_md5:
        return {
            "archived": False,
            "object": target,
            "error": "archive object size or MD5 does not match the local result",
        }
    return {
        "archived": True,
        "object": target,
        "sha256": sha256_file(path),
        "generation": generation,
        "verified_size": remote_size,
        "verified_md5": remote_md5,
    }
