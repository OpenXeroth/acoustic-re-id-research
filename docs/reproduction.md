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

The package name remains `xinyenyana` to preserve imports and the research code's source digest. `docs/provenance.json` records the source snapshot and SHA-256 of every unchanged exported file. The original Git history is not included.

## Run analyses on your own data and compute

1. Obtain the [datasets](datasets.md) and [models](models.md) from their publishers. Keep media, annotations and embeddings outside this repository.
2. Read each command's `--help` and the dataset reader before constructing a benchmark. Source readers and model adapters retain the original `/mnt/data/xinyenyana/` defaults; use supported path options, or reproduce that local directory layout. Do not run original operational scripts blindly.
3. Populate the protocol templates locally where needed. The E01d test fixture contains synthetic identities and cannot reproduce the paper's sample. The templates deliberately omit real identity lists.
4. Retain configuration, code/source hashes, package inventories, model weight digests, seeds, grouped split hashes, predictions and metrics for every run.
5. Configure `XINYENYANA_RESULT_ARCHIVE` to a GCS prefix you control for commands that require an archive. The authors' archive is private. The original CLI's heavy jobs require `XENWARDEN_LEASE`; this is an authors' compute-admission integration, not a public service. For independent infrastructure, call the underlying analysis functions with your scheduler, or implement your own admission adapter. Do not invent an admission receipt.

`uv.lock` pins the base package and optional dependencies declared in `pyproject.toml`. It is **not** a lock for every Bacpipe/AVEX/Transformers/TensorFlow/PyTorch GPU environment used in the experiments. Those model-specific package inventories and weight hashes belong to the original evidence archive. Installing current model packages alone is not an exact reconstruction.

## Rebuild paper tables from archived results

`scripts/paper_v6_build.sh OUTPUT_DIRECTORY` orchestrates the analysis and integrity gates. Its environment variables (`RESULTS`, `HEADLINE`, `V5`, `WEIGHTS`, `SUPPLEMENTAL`, `V5_GATE`, `BAT`, `PY`) identify required result directories and ledgers. Read the script before running it. The evidence gate rejects incomplete or mismatched inventories; it must not be bypassed to generate a publication table.

The numerical result archive and per-animal records are not included in this release. The research machine was unavailable when this public package was prepared. Software verification therefore does not certify a fresh reproduction of the manuscript's numbers. The manuscript and supporting information contain the published aggregate tables; an independently downloadable, redacted numerical evidence bundle remains outstanding.

## Manuscript version

The linked PDF is the author manuscript v7 incorporating the supplied editorial changes on 2 October 2026. It is not labelled as a peer-reviewed published article. The supporting information is the matching v7 repository document. Internal `v6` filenames identify the analysis generation, not an earlier replacement for the current paper.
