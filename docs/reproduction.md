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

This removes the authors' infrastructure dependency. Original recordings and model weights must still be obtained under their publishers' terms; do not redistribute them with this package.

`uv.lock` pins the base package and optional dependencies declared in `pyproject.toml`. It is **not** a lock for every Bacpipe/AVEX/Transformers/TensorFlow/PyTorch GPU environment used in the experiments. The evidence deposit includes three retained package inventories, original model-load provenance and 180 weight-file checksums. Installing current model packages alone is not an exact reconstruction.

## Rebuild paper tables from archived results

`scripts/paper_v6_build.sh OUTPUT_DIRECTORY` orchestrates the analysis and integrity gates. Its environment variables (`RESULTS`, `HEADLINE`, `V5`, `WEIGHTS`, `SUPPLEMENTAL`, `V5_GATE`, `BAT`, `PY`) identify required result directories and ledgers. Read the script before running it. The evidence gate rejects incomplete or mismatched inventories; it must not be bypassed to generate a publication table.

Download the [v10 evidence bundle](https://doi.org/10.5281/zenodo.23174588) separately. The final documents are in `papers/`; `prior-v9/acoustic-re-id-evidence-v9.zip` preserves the follow-up results and the nested original v8 archive. Its README and checksum manifests describe 194 follow-up and 224 original aggregate projections and their original hashes, anonymous split membership, model/environment provenance and the cached-vector audit. Verify its checksums before use. The projections deliberately omit real annotations and per-clip observations, so they cannot be substituted blindly into the historical evidence gate, which checks original bytes and schemas. Do not disable that gate.

For a fresh inference run, reconstruct local annotations using the original publishers' data and the anonymous split membership/audio hashes; retain those real annotations locally. The software does not fetch restricted media, annotations or model weights automatically. The publication audit reused retained vectors and did not repeat all model inference.

## Manuscript version

The linked PDF and supporting information are author manuscript v10, dated 6 October 2026, approved by the authors for bioRxiv submission. The paired-clip audit and funding/interest declarations are complete; the work is not yet submitted or peer reviewed. Internal `v6` and `v7` script names identify their historical analysis generation. See [publication status](publication-status.md).

## Review sensitivity follow-up

The [fixed follow-up protocol](experiments/review-sensitivity.md) defines the new comparisons. Use `scripts/review_sensitivity.py` on locally retained, hash-checked vectors; its `split`, `open` and `paired` modes require an archive prefix and preserve per-run provenance. Full outputs include private annotations and must remain in the restricted result archive. `scripts/project_review_sensitivity.py` validates archive receipts and publishes whitelisted aggregates. Read the [follow-up report](review-sensitivity/README.md) before interpreting these results. No model is selected on these test outcomes.
