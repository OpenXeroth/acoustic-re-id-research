# Reproduction and release limits

## Verify the software

```bash
uv sync --locked --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run python scripts/check_docs.py
uv build
```

The package name remains `xinyenyana` to preserve imports and the research code's source digest. `docs/provenance.json` records the source snapshot and SHA-256 of every unchanged exported file. Modified files are recorded separately with their original export digest and corrected digest. Newly added code has its own hashes. The original Git history is not included.

## Run analyses on your own data and compute

1. Obtain the [datasets](datasets.md) and [models](models.md) from their publishers. Keep media, annotations and embeddings outside this repository.
2. Read each command's `--help` and the dataset reader before constructing a benchmark. Source readers and model adapters retain the original `/mnt/data/xinyenyana/` defaults; use supported path options, or reproduce that local directory layout. Do not run original operational scripts blindly.
3. Populate the protocol templates locally where needed. The E01d test fixture contains synthetic identities and cannot reproduce the paper's sample. The templates deliberately omit real identity lists.
4. Retain configuration, code/source hashes, package inventories, model weight digests, seeds, grouped split hashes, predictions and metrics for every run.
5. On independently managed compute, pass the global `--local-archive /absolute/path/to/results` option before the command. It explicitly selects independent execution, retains source provenance, and writes a verified content-addressed local copy without XenWarden or GCS credentials. Arrange a separate backup of that directory. Without this option, the original managed-compute requirements remain: `XENWARDEN_LEASE` for heavy commands and `XINYENYANA_RESULT_ARCHIVE` for a GCS archive you control. Do not invent an admission receipt.

```bash
uv run xinyenyana --local-archive /absolute/path/to/results run-a6 --help
```

This removes the authors' infrastructure dependency; it does not supply the missing original datasets/manifests, model environments or numerical evidence.

`uv.lock` pins the base package and optional dependencies declared in `pyproject.toml`. It is **not** a lock for every Bacpipe/AVEX/Transformers/TensorFlow/PyTorch GPU environment used in the experiments. Those model-specific package inventories and weight hashes belong to the original evidence archive. Installing current model packages alone is not an exact reconstruction.

## Rebuild paper tables from archived results

`scripts/paper_v6_build.sh OUTPUT_DIRECTORY` orchestrates the analysis and integrity gates. Its environment variables (`RESULTS`, `HEADLINE`, `V5`, `WEIGHTS`, `SUPPLEMENTAL`, `V5_GATE`, `BAT`, `PY`) identify required result directories and ledgers. Read the script before running it. The evidence gate rejects incomplete or mismatched inventories; it must not be bypassed to generate a publication table.

The numerical result archive and per-animal records are not included in this release. The research machine was unavailable when this public package was prepared. Software verification therefore does not certify a fresh reproduction of the manuscript's numbers. The manuscript and supporting information contain the published aggregate tables; an independently downloadable, redacted numerical evidence bundle remains outstanding.

## Manuscript version

The linked PDF and supporting information are author manuscript v8, a corrected draft. The paired-clip error-rate impact audit and author declarations remain outstanding; it is not labelled submission-ready or peer-reviewed. Internal `v6` and `v7` script names identify their historical analysis generation. See [publication status](publication-status.md).
