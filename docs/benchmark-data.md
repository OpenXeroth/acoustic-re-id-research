# Building benchmark inputs

Obtain recordings and annotations from the original [dataset publishers](datasets.md), subject to their licences. Keep recordings, per-animal annotations and restricted derived embeddings outside Git.

Use the relevant reader and each CLI command's `--help` to determine input layout. `src/xinyenyana/cli.py` maps endpoint names to their readers. Dataset builders retain input hashes and selection provenance; preserve their summaries and manifests alongside results.

The public experiment configurations omit real identity lists. The E01d fixture is synthetic and must not be used to claim reproduction of the paper. The older 16-bird great-tit and zebra-finch construction manifests are not included, and the public release does not contain a complete reconstruction of their preparation. Obtain the original manifests before attempting those exact benchmarks; do not infer their sample from the supplementary table.

The larger great-tit reader, BirdPark builder, bat builder and rook readers are included, but their availability alone does not establish an end-to-end reproduction. Published labels may denote nests, territories or tags rather than directly observed individual callers. Preserve these distinctions and each recording/session grouping.

The original `/mnt/data/xinyenyana/` and `gs://xinyenyana/` paths identify the authors' infrastructure, not public download locations. Use local path options and the independent-compute mode documented in [reproduction](reproduction.md).
