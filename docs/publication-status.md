# Author manuscript v10: publication status

Version 10, 6 October 2026, incorporates the final editorial corrections and is approved by the authors for bioRxiv submission. It has not yet been submitted or peer reviewed. Previous versions remain available.

## Final editorial pass

- Corrected the abstract and opening Discussion to describe substantial background identity information, with balanced accuracy 0.714 versus 0.791 for calls, without implying equivalence.
- Added the zebra-finch timing caveat to the Discussion and used “random split of individual clips” consistently.
- Qualified the abstract’s 46% as the highest observed median across 36 networks and two scorers, with four enrolled birds.
- Simplified the software-licence sentence and retained the agreed CC BY-NC terms in archive and submission metadata.
- Adopted the illustrated graphical abstract with matching timing, data-availability and open-set qualifications.
- No experiments, aggregate results or analysis software changed. Software remains version 0.3.0.

## Review follow-up

- Completed and independently archived all 194 new result files: six count-matched comparisons, 185 open-set model/endpoint outputs (including five explicitly uncounted duplicate checkpoints), and three descriptive timing controls.
- All 18 count-matched classifier comparisons retain the split effect (mean differences 0.196–0.508). The revised headline states the clip split unit and intended new-session task.
- Tested both open-set scorers with independent calibration; the highest median on wild session-disjoint endpoints changes from 37.4% to 46.2%. Neither is a performance ceiling.
- Tested 4/8/12 calibration strangers without changing galleries or test roles. More calibration strangers did not consistently resolve transfer of the threshold.
- Reported balanced background accuracy and separate majority references. BirdPark timing controls within seven days are unavailable in the retained test clips; fixed-donor rotation is explicitly unperformed.
- Updated native manuscripts, figures, supporting information and public explanations. See the [complete follow-up report](review-sensitivity/README.md) and [fixed protocol](experiments/review-sensitivity.md).

## Previous verification audit retained

- Corrected tied-score verification and excluded invalid raw magnitude controls from cosine-only analyses. The valid kernel-ridge controls remain.
- Recomputed all 19 retained BirdNET headline/rook variants from original cached embeddings. Manifest, query order and pair summaries match; per-query correctness and classifier AUC reproduce exactly. All paired EER values in Tables S3/S4 and classifier EERs are unchanged. Some true-accept rates differ only by floating-point rounding below 10⁻¹⁵. Full audit: [evidence DOI](https://doi.org/10.5281/zenodo.23159989).
- Checked all 224 required original results against their content hashes and independent GCS object sizes/MD5 values: 224/224 match.
- Recovered all 108 sweep/refresh/replay source digests. Their initial sweep diagnostics use 999 permutations and 2,000 bootstrap draws; the manuscript now distinguishes these from the 9,999/10,000 headline/background analyses and 10,000 paired-comparison draws.
- Confirmed 11 whales, 162 enrolment and 72 test calls, with paired noise counts, in all 11 original frequency-shift manifests.
- Released aggregate projections of the 224 results, anonymous split membership (including the 16-bird great-tit and zebra-finch cohorts), three retained package inventories, model provenance and weight hashes. Media, embeddings, real identity annotations and per-clip observations are excluded.
- Established that the retained BirdNET v3 preview ONNX file differs from Zenodo 18247420 preview3. The paper identifies the actual Bacpipe-supplied file and its SHA-256 instead.
- Confirmed funding and interests: Xeroth AI Limited funded the research; Graham Wallington is CEO, Jackie Lighten is COO, and both are shareholders. Their roles in the work are disclosed.
- Assigned version-specific archive citations: [software 0.2.1](https://doi.org/10.5281/zenodo.23159972) and [v8 evidence](https://doi.org/10.5281/zenodo.23159989). Manuscript/evidence documentation: CC BY-NC 4.0; existing software: Apache-2.0; third-party terms unchanged.

## Current archives

[Software 0.3.0](https://doi.org/10.5281/zenodo.23167850) and [v10 evidence](https://doi.org/10.5281/zenodo.23174588) contain the review follow-up and final publication documents. Earlier version-specific archives remain unchanged.

## Final author step

The final editorial conditions have been addressed and author sign-off is recorded. Graham can submit the prepared main manuscript and separate supporting information through his bioRxiv account, then inspect and approve the portal-generated proof. No preprint DOI exists until bioRxiv assigns it; that DOI must subsequently be added to the repository and research page.

The audit covers the reported paired EERs and headline verification diagnostics. The shared classifier AUC calculation is unchanged by the software correction. Historical unused classifier EER/TAR diagnostics in the aggregate archive have not all been recomputed; they must not be described as newly validated results. Open-set statistics use a separate implementation. The previous audit did not perform new model inference. The new follow-up analyses reuse frozen embeddings; they are described separately above.

Djuma/BirdNET-Cloud geography remains separate from this manuscript release. Portal submission/proof approval and the subsequent bioRxiv DOI link remain the final steps. A planned submission date is not a guaranteed posting date: bioRxiv screens submitted manuscripts.
