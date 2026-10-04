# Models

The paper compares 36 neural networks plus duration-only and level-only controls. The registries in `src/xinyenyana/a5.py` and `bioacoustic.py` are authoritative for names and adapters. Model weights are not included or relicensed. Check each publisher’s terms before downloading or using outputs.

## BirdNET and animal/general audio models

- [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer): v2.4 baseline; package `birdnet==1.1.0`. BirdNET model licensing includes non-commercial restrictions.
- [Bacpipe](https://github.com/bioacoustic-ai/bacpipe): adapters for Perch 2, Perch v1, BirdNET v3 preview, SurfPerch, AvesEcho, AudioProtoPNet, ConvNeXt BirdSet, Bird-MAE, ProtoCLR, RCL, BirdAVES, AVES, NatureBEATs, BioLingual, BEATs, Audio-MAE and VGGish.
- [Earth Species Project AVEX](https://github.com/earthspecies/avex): `esp_aves2_sl_beats_all` and `esp_aves2_effnetb0_all`.

## Speech models

- [espnet/xeus](https://huggingface.co/espnet/xeus): the original adapter expects a separately downloaded checkpoint; recover its recorded weight digest before exact reproduction.
- [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
- [facebook/wav2vec2-large-robust](https://huggingface.co/facebook/wav2vec2-large-robust)
- [facebook/wav2vec2-conformer-rope-large](https://huggingface.co/facebook/wav2vec2-conformer-rope-large)
- [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- [facebook/mms-300m](https://huggingface.co/facebook/mms-300m)
- [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960)
- [facebook/hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k)
- [facebook/data2vec-audio-base-100h](https://huggingface.co/facebook/data2vec-audio-base-100h)
- [facebook/data2vec-audio-base-960h](https://huggingface.co/facebook/data2vec-audio-base-960h)
- [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)
- [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large)
- [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv)
- [microsoft/unispeech-sat-base-plus](https://huggingface.co/microsoft/unispeech-sat-base-plus)
- [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [speechbrain/spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)
- [speechbrain/spkrec-resnet-voxceleb](https://huggingface.co/speechbrain/spkrec-resnet-voxceleb)

The base uv.lock is not the complete GPU inference environment. Exact model load recipes, package inventories and weight digests must accompany any numerical reproduction; see [reproduction](reproduction.md).
