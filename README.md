# Acoustic re-identification research

**Can a computer recognise an individual animal by its voice—or is it recognising the recording?**

Research code accompanying *Recording context and data splitting can inflate acoustic re-identification accuracy across species and studies*, by Graham Wallington and Jackie Lighten (2026, author manuscript v10, 6 October 2026).

[Research website](https://open.xeroth.ai/acoustic-re-id-research/) · [Full paper (PDF)](https://open.xeroth.ai/papers/acoustic-re-id-v10.pdf) · [Supporting information](docs/supporting-information.md) · [Datasets](docs/datasets.md) · [Models](docs/models.md) · [Review follow-up results](docs/review-sensitivity/README.md)

We compare 36 pretrained neural networks and two simple controls across 13 datasets covering nine species. Background sound, recording sessions and data splitting can make individual recognition look more reliable than it is. The code tests these shortcuts and evaluates the harder task of recognising familiar animals while rejecting strangers. Djuma recordings are not part of this benchmark.

**Verification complete:** all 224 required result files match their independent archive copies. The 19-run cached-vector audit reproduces classification, AUC and the reported paired-clip error rates. The authors have approved the manuscript for bioRxiv submission. See [publication status](docs/publication-status.md).

[Software DOI: 10.5281/zenodo.23167850](https://doi.org/10.5281/zenodo.23167850) · [Evidence DOI: 10.5281/zenodo.23174588](https://doi.org/10.5281/zenodo.23174588)

## Start here

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/OpenXeroth/acoustic-re-id-research.git
cd acoustic-re-id-research
uv sync --locked --extra dev
uv run pytest
uv run xinyenyana --help
```

These software tests use synthetic fixtures and do not download recordings or run live ingestion. Installing dependencies needs internet access. GPU inference requires additional, model-specific environments and separately downloaded weights.

## What is included

- `src/xinyenyana/`: the analysis package, with dataset readers, model adapters, grouped evaluation, recording/background controls, open-set evaluation and provenance checks.
- `scripts/`: the original figure/table analysis and evidence checks. Historical `v6` filenames remain because v7 uses those analyses.
- `tests/`: the original offline tests, with a synthetic E01d protocol fixture.
- `uv.lock`: the original base dependency lock.
- `configs/experiments/`: protocol templates, with animal identity lists omitted.

See [reproduction and current limits](docs/reproduction.md), [source provenance](docs/provenance.json), [contributing](CONTRIBUTING.md) and the [documentation index](docs/index.md).

The separate evidence deposit contains aggregate projections of all 224 required results, anonymous split membership, original environments and weight inventories, source checksums and the verification audit. Projections exclude real identity annotations, per-clip observations, recordings and embeddings. They are not byte-identical substitutes for the original records. A fresh model-inference run requires original datasets and weights from their publishers and locally reconstructed annotations; the publication audit reused retained vectors rather than rerunning inference.

## Licence and citation

Original research code is released under [Apache-2.0](LICENSE). Third-party data, models and libraries retain their own licences, which may restrict commercial use. The manuscript, supporting information and original evidence documentation are CC BY-NC 4.0: attributed sharing and adaptation for non-commercial purposes. Software remains Apache-2.0, including its commercial-use permission. Third-party terms are unaffected. See [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff).

NatureCam is part of [Xeroth](https://xeroth.ai). [OpenXeroth](https://open.xeroth.ai/) shares its research and open-source work.
