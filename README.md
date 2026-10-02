# Acoustic re-identification research

**Can a computer recognise an individual animal by its voice—or is it recognising the recording?**

Research code accompanying *Recording context and data splitting can inflate acoustic re-identification accuracy across species and studies*, by Graham Wallington and Jackie Lighten (2026, author manuscript v7).

[Research website](https://open.xeroth.ai/acoustic-re-id-research/) · [Full paper (PDF)](https://open.xeroth.ai/papers/acoustic-re-id-v7.pdf) · [Supporting information](docs/supporting-information.md) · [Datasets](docs/datasets.md) · [Models](docs/models.md)

We compare 36 pretrained neural networks and two simple controls across 13 datasets covering nine species. Background sound, recording sessions and data splitting can make individual recognition look more reliable than it is. The code tests these shortcuts and evaluates the harder task of recognising familiar animals while rejecting strangers. Djuma recordings are not part of this benchmark.

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

This release publishes the code, not the private result archive, recordings or weights. The paper's complete numerical analysis has not been rerun for this release. Per-animal annotations and private operational material are excluded. Public dataset access is through the original publishers.

## Licence and citation

Original research code is released under [Apache-2.0](LICENSE). Third-party data, models and libraries retain their own licences, which may restrict commercial use. The manuscript and supporting information are author publications; the software licence does not relicense them. See [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff).

NatureCam is part of [Xeroth](https://xeroth.ai). [OpenXeroth](https://open.xeroth.ai/) shares its research and open-source work.
