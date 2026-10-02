#!/usr/bin/env python3
"""Put the labelled benchmark archives into GCS with the publisher's checksum.

Two things are proved for every file, and both are recorded in the manifest:

    publisher checksum == checksum of the bytes on disk  (the download is right)
    local crc32c       == crc32c stored on the object    (the upload is right)

A GCS composite object carries no md5, so crc32c is what proves the upload.

Downloads are explicit ranged GETs whose Content-Range must start at the
current file length, so a dropped connection resumes exactly where it stopped
and a server that ignores Range cannot corrupt the file. That is the failure
this script exists to remove: the previous fetcher used `curl -C -` against a
source that intermittently ignored Range, which left two archives at exactly
the published size with the wrong md5, and its retry loop deleted only files
larger than expected, so those two could never clear.

Usage (any number of dataset or file-name prefixes; all of them if none):

    python3 scripts/fetch_benchmarks.py rookid
    XINYENYANA_BENCHMARK_CACHE=/mnt/data/xinyenyana/benchmarks \\
      python3 scripts/fetch_benchmarks.py stowell bird-aiid

Media and archives never enter Git. The cache directory is local scratch; the
canonical copy is the GCS object named in the manifest.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

CACHE = os.environ.get("XINYENYANA_BENCHMARK_CACHE", "data/raw/benchmarks")
GCS = os.environ.get("XINYENYANA_BENCHMARK_GCS", "gs://xinyenyana/benchmarks")
MANIFEST = os.environ.get("XINYENYANA_BENCHMARK_MANIFEST", "benchmark-manifest.json")
CHUNK = 16 << 20
# How many ranged GETs are in flight at once. The right number is a property of
# the source, not of this script, so it is settable per run. One Zenodo
# connection from xen1 measured 216 kB/s and four gave about three times that.
# A short Figshare probe on 16 MiB chunks made one connection look six times
# faster than four, and a sustained download contradicted it: four streams held
# 0.53 MB/s over eight minutes and one held 0.28 MB/s over twelve, so the probe
# was measuring something other than sustained rate and the default stands.
STREAMS = int(os.environ.get("XINYENYANA_BENCHMARK_STREAMS", "4"))
MAX_STALL = 20


def zenodo(record: int, key: str) -> str:
    return f"https://zenodo.org/api/records/{record}/files/{key}/content"


# dataset directory, file name, bytes, algorithm, publisher checksum, source URL.
# Sizes and checksums are the publishers' own, read from their APIs.
FILES: list[tuple[str, str, int, str, str, str]] = [
    (
        "stowell-2019-zenodo-1413495",
        "README.txt",
        2954,
        "md5",
        "cc0f3bfc00c50e7fa2b8dd7da43efcde",
        zenodo(1413495, "README.txt"),
    ),
    (
        "stowell-2019-zenodo-1413495",
        "csv.zip",
        50553,
        "md5",
        "99c6d4b24306f079cbe1f5fe457bf7b4",
        zenodo(1413495, "csv.zip"),
    ),
    (
        "stowell-2019-zenodo-1413495",
        "littleowl-fg.zip",
        77137238,
        "md5",
        "9722aebad9fd8c05ccb2a600989a0f18",
        zenodo(1413495, "littleowl-fg.zip"),
    ),
    (
        "stowell-2019-zenodo-1413495",
        "littleowl-bg.zip",
        223118519,
        "md5",
        "27a2016050e7d3e259f7763ad71f4960",
        zenodo(1413495, "littleowl-bg.zip"),
    ),
    (
        "stowell-2019-zenodo-1413495",
        "pipit-fg.zip",
        426612362,
        "md5",
        "53f98cfcce4aaaa952d1d7c91792f208",
        zenodo(1413495, "pipit-fg.zip"),
    ),
    (
        "stowell-2019-zenodo-1413495",
        "pipit-bg.zip",
        603369200,
        "md5",
        "f92789da7748f1302a2d8be5523e5c33",
        zenodo(1413495, "pipit-bg.zip"),
    ),
    (
        "stowell-2019-zenodo-1413495",
        "chiffchaff-fg.zip",
        2709892748,
        "md5",
        "4d1409b783e7deaa55882a69c30a2e23",
        zenodo(1413495, "chiffchaff-fg.zip"),
    ),
    (
        "stowell-2019-zenodo-1413495",
        "chiffchaff-bg.zip",
        4959816888,
        "md5",
        "c220cd97c902564514058a3713ecdf17",
        zenodo(1413495, "chiffchaff-bg.zip"),
    ),
    (
        "bird-aiid-zenodo-17576155",
        "dataset.zip",
        705988754,
        "md5",
        "48866e6aeada3db675e0297e86013f77",
        zenodo(17576155, "dataset.zip"),
    ),
    (
        "rookid-zenodo-6091940",
        "split_data.py",
        784,
        "md5",
        "73073029753824de943e9900771e1fe6",
        zenodo(6091940, "split_data.py"),
    ),
    (
        "rookid-zenodo-6091940",
        "split_data.R",
        1376,
        "md5",
        "befec106fa0f265f522b53922cd90b43",
        zenodo(6091940, "split_data.R"),
    ),
    (
        "rookid-zenodo-6091940",
        "splitting.csv",
        5241,
        "md5",
        "059e626eecec539a11b5844e85bf658e",
        zenodo(6091940, "splitting.csv"),
    ),
    (
        "rookid-zenodo-6091940",
        "RookID.zip",
        22091554631,
        "md5",
        "aec294923cee2b2b7ae20a09825b8f45",
        zenodo(6091940, "RookID.zip"),
    ),
    (
        "birdpark-zenodo-13144875",
        "README.pdf",
        59467,
        "md5",
        "bf9ed8effd1c99910a93e0ca95c06522",
        zenodo(13144875, "README.pdf"),
    ),
    (
        "birdpark-zenodo-13144875",
        "segments.h5",
        1905721,
        "md5",
        "be900f572084a3d83efd6979e3739fe0",
        zenodo(13144875, "segments.h5"),
    ),
    (
        "birdpark-zenodo-13144875",
        "Attachments.zip",
        292640,
        "md5",
        "5d787103db2412bf8adb6a752d65597c",
        zenodo(13144875, "Attachments.zip"),
    ),
    (
        "birdpark-zenodo-13144875",
        "Data.zip",
        12335536947,
        "md5",
        "32d1ae6049556c803f68b6d354c952ca",
        zenodo(13144875, "Data.zip"),
    ),
    (
        "zebra-finch-figshare-11905533",
        "AdultVocalizations.zip",
        140645384,
        "md5",
        "696bb91ca90ea116468bbce85a67aa0f",
        "https://ndownloader.figshare.com/files/21833430",
    ),
    (
        "wytham-great-tit-osf-n8ac9",
        "main.csv",
        710595,
        "sha256",
        "497cc05221f93cef597f95390ea7a8122cd42d7da5dba6cc9ed0dab3d475f4d9",
        "https://osf.io/download/tnj2y/",
    ),
    (
        "wytham-great-tit-osf-n8ac9",
        "morphometrics.csv",
        3716332,
        "sha256",
        "67953ccd0a54dcd7a786b64bd0ab9ec347e40a4ef34a4c356ea2d12d7588ec4d",
        "https://osf.io/download/kg32a/",
    ),
    (
        "wytham-great-tit-osf-n8ac9",
        "great-tit-hits.csv",
        111502303,
        "sha256",
        "580fee8440d7e213b7c8507903818d7748efaf4052ed94740a25ec8c1e5ccb80",
        "https://osf.io/download/9a56y/",
    ),
    (
        "wytham-great-tit-osf-n8ac9",
        "song-files.zip.part1",
        3655144225,
        "sha256",
        "b10173d9810aba0049755c7cb0167ebfe3903b95dfb08984d9a7c6556357037c",
        "https://osf.io/download/xywcm/",
    ),
    (
        "wytham-great-tit-osf-n8ac9",
        "song-files.zip.part2",
        3735396385,
        "sha256",
        "ad9aab5253bf7a8f70c32db17b3e92f5dfb3df18581a080c7f29e01bb6b36ec5",
        "https://osf.io/download/5ytvd/",
    ),
    (
        "wytham-great-tit-osf-n8ac9",
        "song-files.zip.part3",
        3741555821,
        "sha256",
        "caaa0a78de733ac83ababa1b9d8e1cc977ee34abc9e442b0aaf609bb28261bb9",
        "https://osf.io/download/cez45/",
    ),
    (
        "wytham-great-tit-osf-n8ac9",
        "song-files.zip.part4",
        248315109,
        "sha256",
        "2cc2fdc7da1e76ed60d3217a251404e916b810973620de7cbdd3a570bd2d444d",
        "https://osf.io/download/6x243/",
    ),
    (
        "rook-nest-calls-zenodo-15609846",
        "rook_vocal_signature-main.zip",
        71510,
        "md5",
        "a24def1f04ddda9b9190ee3134701ef6",
        zenodo(15609846, "rook_vocal_signature-main.zip"),
    ),
    (
        "rook-nest-calls-zenodo-15609846",
        "supplementary.zip",
        12156315,
        "md5",
        "0ac3494450168888f4424fb47acc0224",
        zenodo(15609846, "supplementary.zip"),
    ),
    # The Egyptian fruit bat corpus is published as one annotation table, one
    # file-information table and about 224 archives of roughly three gigabytes
    # each. The two tables say which archive holds which vocalisation and who
    # emitted it, so they are fetched first and the archives that a built
    # endpoint needs are added to this table afterwards.
    (
        "prat-2017-figshare-c3666502",
        "Annotations.csv",
        3286405,
        "md5",
        "7781a3725d6f2277de74da84b1383ce0",
        "https://ndownloader.figshare.com/files/7379008",
    ),
    (
        "prat-2017-figshare-c3666502",
        "FileInfo.csv",
        31574701,
        "md5",
        "b27252490ac5618bd55a9038110809d5",
        "https://ndownloader.figshare.com/files/8900695",
    ),
    (
        "prat-2017-figshare-c3666502",
        "Metadata.pdf",
        367257,
        "md5",
        "d9d1c2dcc0f687fd3b5c98e76e1383ab",
        "https://ndownloader.figshare.com/files/8900800",
    ),
    # The archives holding the recordings this project needs. Six bats were
    # recorded in treatments 16 to 19, from December 2012 to June 2013, and
    # again in treatment 20 in February 2014. Those recordings are spread over
    # files210 to files224 and nothing outside that range is fetched. The rest
    # of the corpus is about 200 more archives of the same size.
    (
        "prat-2017-figshare-c3666502",
        "files210.zip",
        3049199327,
        "md5",
        "9c01a0622b2dde0a3d3686ba0263eec3",
        "https://ndownloader.figshare.com/files/8879683",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files211.zip",
        3199079432,
        "md5",
        "0abad58bd6e7015868223a5e11259c48",
        "https://ndownloader.figshare.com/files/8879179",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files212.zip",
        3036142257,
        "md5",
        "8981e045f7648b2a7c8be53e7667b0d4",
        "https://ndownloader.figshare.com/files/8879287",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files213.zip",
        2959680169,
        "md5",
        "e8dc3c504e75b1ec6a5bbb8107f65d27",
        "https://ndownloader.figshare.com/files/8879659",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files214.zip",
        2991098213,
        "md5",
        "6daef343c7fdb806fa0410839553e90f",
        "https://ndownloader.figshare.com/files/8879674",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files215.zip",
        2868671367,
        "md5",
        "eed7a1c35d6e35ccb610f39357f67008",
        "https://ndownloader.figshare.com/files/8879662",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files216.zip",
        2814995134,
        "md5",
        "a258f348e6931ee58051d074b34dd7f5",
        "https://ndownloader.figshare.com/files/8879641",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files217.zip",
        2816018581,
        "md5",
        "0af9cb211e64120b36f397f7ff0fd3d1",
        "https://ndownloader.figshare.com/files/8879632",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files218.zip",
        2818492494,
        "md5",
        "8ad7b0b06f6f961c0ca3280b0d7b6e41",
        "https://ndownloader.figshare.com/files/8879653",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files219.zip",
        2780818280,
        "md5",
        "9dfa56bafa4d0526a72c210f5c418435",
        "https://ndownloader.figshare.com/files/8879617",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files220.zip",
        2903298982,
        "md5",
        "a8bf9eb1a4b8b99b3ada6a94570da0ee",
        "https://ndownloader.figshare.com/files/8879623",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files221.zip",
        2962658616,
        "md5",
        "538b9dd5831577ac66a464f7855e9946",
        "https://ndownloader.figshare.com/files/8879611",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files222.zip",
        3001287451,
        "md5",
        "b7727411e0a185350cd217e4f5aadda4",
        "https://ndownloader.figshare.com/files/8879608",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files223.zip",
        3038419056,
        "md5",
        "f6c8dbae9dea20e4d4e2944537480c86",
        "https://ndownloader.figshare.com/files/8879599",
    ),
    (
        "prat-2017-figshare-c3666502",
        "files224.zip",
        3668188044,
        "md5",
        "327f70b070ed295b636af178bac3a972",
        "https://ndownloader.figshare.com/files/8879602",
    ),
]

# The bulk archives of the two re-opened wild endpoints are not here. They are
# too large to stage on a disk. Fetching those needs a streaming path into
# GCS instead; its own table holds them. Still absent, and why:
#   rook-nest-calls-zenodo-15609846/intermediate_results.zip (803,655,509
#     bytes) holds the source authors' analysis outputs, not recordings.
#   monk-parakeet raw_data_20.zip (36,961,539,480 bytes) holds the 2020
#     recordings; the screened 16 identities came from 2021.
# Adding either is one row in this table.


def log(*parts: object) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *parts, flush=True)


def local_checksum(path: str, algo: str) -> str:
    digest = hashlib.new(algo)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def local_crc32c(path: str) -> str | None:
    """crc32c of a local file, base64, as GCS reports it. gsutil carries the
    compiled implementation, so this does not add a Python dependency."""
    result = subprocess.run(["gsutil", "hash", "-c", path], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "crc32c" in line:
            return line.split(":")[-1].strip()
    return None


def ranged_get(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 206:
            raise OSError(f"expected 206, got {response.status}")
        served = int(response.headers["Content-Range"].split()[1].split("-")[0])
        if served != start:
            raise OSError(f"content-range starts at {served}, asked for {start}")
        body = response.read()
    if len(body) != end - start + 1:
        raise OSError(f"short read: {len(body)} of {end - start + 1} bytes")
    return body


def download(url: str, path: str, size: int) -> None:
    """Resumable ranged download; STREAMS chunks in flight, appended in order."""
    part = path + ".part"
    stall = 0
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= size:
            break
        spans = []
        for index in range(STREAMS):
            start = have + index * CHUNK
            if start >= size:
                break
            spans.append((start, min(start + CHUNK, size) - 1))
        try:
            with concurrent.futures.ThreadPoolExecutor(len(spans)) as pool:
                blocks = list(pool.map(lambda span: ranged_get(url, span[0], span[1]), spans))
            with open(part, "ab") as handle:
                for block in blocks:
                    handle.write(block)
            stall = 0
        except Exception as error:  # noqa: BLE001 - any transport failure is retried
            stall += 1
            log(
                "  range error",
                os.path.basename(path),
                have,
                type(error).__name__,
                str(error)[:120],
                "stall",
                stall,
            )
            if stall >= MAX_STALL:
                raise
            time.sleep(min(60, 5 * stall))
    os.replace(part, path)


def gcs_stat(uri: str) -> dict[str, str] | None:
    result = subprocess.run(["gsutil", "stat", uri], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    fields = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def main() -> int:
    selectors = sys.argv[1:]
    manifest: list[dict[str, object]] = []
    if os.path.exists(MANIFEST):
        manifest = json.load(open(MANIFEST))
    settled = {(row["dataset"], row["name"]) for row in manifest if row.get("status") == "ok"}
    failures = 0

    for dataset, name, size, algo, published, url in FILES:
        if selectors and not any(dataset.startswith(s) or name.startswith(s) for s in selectors):
            continue
        if (dataset, name) in settled:
            continue
        directory = os.path.join(CACHE, dataset)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        uri = f"{GCS}/{dataset}/{name}"
        row: dict[str, object] = {
            "dataset": dataset,
            "name": name,
            "bytes": size,
            "algo": algo,
            "publisher_checksum": published,
            "source_url": url,
            "gcs": uri,
        }

        for attempt in (1, 2):
            if not (os.path.exists(path) and os.path.getsize(path) == size):
                log("download", dataset, name, size, "attempt", attempt)
                try:
                    download(url, path, size)
                except Exception as error:  # noqa: BLE001 - recorded, not raised
                    log("  download failed", name, str(error)[:200])
                    row["status"] = "download_failed"
                    break
            row["local_checksum"] = local_checksum(path, algo)
            if row["local_checksum"] == published:
                log("verified", dataset, name, algo, published)
                row["status"] = "verified_local"
                break
            log(
                "  CHECKSUM MISMATCH",
                dataset,
                name,
                "publisher",
                published,
                "local",
                row["local_checksum"],
                "attempt",
                attempt,
            )
            os.remove(path)
            row["status"] = "checksum_mismatch"

        if row.get("status") != "verified_local":
            failures += 1
            manifest.append(row)
            json.dump(manifest, open(MANIFEST, "w"), indent=2)
            continue

        row["local_crc32c"] = local_crc32c(path)
        stat = gcs_stat(uri)
        if stat is None or stat.get("Content-Length") != str(size):
            log("upload", uri)
            upload = subprocess.run(
                [
                    "gsutil",
                    "-q",
                    "-o",
                    "GSUtil:parallel_composite_upload_threshold=150M",
                    "cp",
                    path,
                    uri,
                ]
            )
            if upload.returncode != 0:
                log("  upload failed", uri)
                row["status"] = "upload_failed"
                failures += 1
                manifest.append(row)
                json.dump(manifest, open(MANIFEST, "w"), indent=2)
                continue
            stat = gcs_stat(uri)
        stat = stat or {}
        row["stored_crc32c"] = stat.get("Hash (crc32c)")
        row["stored_bytes"] = int(stat.get("Content-Length", -1))
        row["component_count"] = stat.get("Component-Count")
        row["status"] = (
            "ok"
            if row["stored_crc32c"] == row["local_crc32c"] and row["stored_bytes"] == size
            else "gcs_mismatch"
        )
        if row["status"] != "ok":
            failures += 1
        log(row["status"], uri, "crc32c local", row["local_crc32c"], "stored", row["stored_crc32c"])
        manifest.append(row)
        json.dump(manifest, open(MANIFEST, "w"), indent=2)

    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    log("run finished", " ".join(selectors) or "all", "failures", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
