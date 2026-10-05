# Author manuscript v8: publication status

Version 8, 5 October 2026, is prepared for final author approval before bioRxiv submission. It is not peer reviewed. The original v7 and corrected v8-draft files remain available.

## Completed

- Corrected tied-score verification and excluded invalid raw magnitude controls from cosine-only analyses. The valid kernel-ridge controls remain.
- Recomputed all 19 retained BirdNET headline/rook variants from original cached embeddings. Manifest, query order and pair summaries match; per-query correctness and classifier AUC reproduce exactly. All paired EER values in Tables S3/S4 and classifier EERs are unchanged. Some true-accept rates differ only by floating-point rounding below 10⁻¹⁵. Full audit: [evidence DOI](https://doi.org/10.5281/zenodo.23159989).
- Checked all 224 required original results against their content hashes and independent GCS object sizes/MD5 values: 224/224 match.
- Recovered all 108 sweep/refresh/replay source digests. Their initial sweep diagnostics use 999 permutations and 2,000 bootstrap draws; the manuscript now distinguishes these from the 9,999/10,000 headline/background analyses and 10,000 paired-comparison draws.
- Confirmed 11 whales, 162 enrolment and 72 test calls, with paired noise counts, in all 11 original frequency-shift manifests.
- Released aggregate projections of the 224 results, anonymous split membership (including the 16-bird great-tit and zebra-finch cohorts), three retained package inventories, model provenance and weight hashes. Media, embeddings, real identity annotations and per-clip observations are excluded.
- Established that the retained BirdNET v3 preview ONNX file differs from Zenodo 18247420 preview3. The paper identifies the actual Bacpipe-supplied file and its SHA-256 instead.
- Confirmed funding and interests: Xeroth AI Limited funded the research; Graham Wallington is CEO, Jackie Lighten is COO, and both are shareholders. Their roles in the work are disclosed.
- Assigned version-specific archive citations: [software 0.2.1](https://doi.org/10.5281/zenodo.23159972) and [v8 evidence](https://doi.org/10.5281/zenodo.23159989). Manuscript/evidence documentation: CC BY-NC 4.0; existing software: Apache-2.0; third-party terms unchanged.

## Final author step

Jackie must approve the final main manuscript, separate supporting information, declarations and submission. Graham can then submit the prepared files through his bioRxiv account. No preprint DOI exists until bioRxiv assigns it; that DOI must subsequently be added to the repository and research page.

The audit covers the reported paired EERs and headline verification diagnostics. The shared classifier AUC calculation is unchanged by the software correction. Historical unused classifier EER/TAR diagnostics in the aggregate archive have not all been recomputed; they must not be described as newly validated results. Open-set statistics use a separate implementation. No new model inference or additional scientific experiment is claimed.

Djuma/BirdNET-Cloud geography and additional experimental comparisons remain separate from this manuscript release.
