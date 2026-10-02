# Dataset sources

Download recordings and annotations from the original publishers, under their terms. Dataset names in the paper sometimes describe different splits or subsets of the same source release; thirteen evaluation datasets do not mean thirteen independent repositories.

| Source | Animals / benchmark role | Publisher |
|---|---|---|
| Stowell release | Chiffchaff, little owl and tree pipit, including paired backgrounds and within/across-year splits | [Zenodo 1413495](https://doi.org/10.5281/zenodo.1413495) |
| Huang release, version 3 | Red-tailed black cockatoo, little penguin and published embeddings | [Zenodo 17576155](https://doi.org/10.5281/zenodo.17576155) |
| RookID | Rooks, shared recordings and repeated days/channels | [Zenodo 6091940](https://doi.org/10.5281/zenodo.6091940) |
| Zebra finch adult vocalisations | Captive zebra finches | [Figshare 11905533](https://doi.org/10.6084/m9.figshare.11905533) |
| Wytham great tit | Nestbox-attributed recordings and ring metadata | [OSF N8AC9](https://doi.org/10.17605/OSF.IO/N8AC9) |
| Egyptian fruit bat | Vocalisations, emitter attribution and recording treatments | [Figshare collection 3666502](https://doi.org/10.6084/m9.figshare.c.3666502) |
| BirdPark | Additional benchmark/control analyses in the code | [Zenodo 13144875](https://doi.org/10.5281/zenodo.13144875) |

The code also contains historical [North Atlantic right whale](https://github.com/avokloti/narw-acoustic-identification) diagnostics. That source did not declare a redistribution licence when evaluated; no copy is distributed here and it is not one of the paper's nine species.

Identity evidence varies across datasets, from observed or ringed animals to nest/territory attribution. Treat these differences as limitations, not equivalent ground truth. Keep splits grouped by individual, session, source recording and detection bout as the protocol requires. Random clip splits in this research are deliberate shortcut diagnostics, not a recommended evaluation design.
