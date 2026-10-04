# Author manuscript v8: correction status

This is a corrected author draft, not a bioRxiv submission or a peer-reviewed article.
The original v7 release remains available under `paper-v7-2026-10-02`.

## Completed in this revision

- Excluded the raw duration and loudness controls from Tables S10 and S15 and their cosine-only analysis paths. The valid kernel-ridge control results remain. Table renderers and summary counts apply the same exclusions, including when reading historical archives.
- Corrected tied-score handling in paired verification and classifier verification diagnostics. Equal scores enter together at attainable thresholds; low-FAR acceptance cannot be obtained by ordering tied observations using their true labels. Added regression tests. This software correction is not a numerical rerun of the paper.
- Reported both fruit-bat correlations and the dependence on the session-mean settings; clarified preprocessing, splits, standardisation, bootstrap p values and evidence-gate scope.
- Corrected the availability statement, linked the public release, and described internal pre-specification accurately.
- Added explicit independent-compute execution with a local content-addressed archive, without a fabricated XenWarden receipt. The managed-compute defaults remain intact.
- Added the missing public documentation entry points and XEUS model link. The public protocol summary is retrospective documentation, not evidence of prospective registration.

## Required before submission

1. Recompute the paired-clip equal error rates in Tables S3 and S4 from the archived embeddings/scores, and audit all uses of `evaluation.verification_metrics` for tied-score effects. Retain the source hashes, original and corrected values and the new code digest. The draft labels the historical paired-clip error rates as awaiting verification. Do not treat passing software tests as this check.
2. Confirm the actual funding, each author's employment/ownership or other financial interests, and the funder's role. Replace the pending declaration with factual text; do not assert that the company had no role without evidence.
3. Obtain all authors' consent to this corrected version and agree the bioRxiv reuse licence. Then export the final main PDF and separate supporting-information PDF with the same version label.
4. Review the original corrected-sweep, speech, selection-refresh and speech-replay records for resampling counts where the original evidence gate did not enforce them. Recheck the right-whale sample count from its original manifest.

The research host was unreachable and the available archive credentials did not permit an exact-object read during preparation. These are access limits, not evidence that archived results are missing. The score audit and record checks can run on a CPU once the relevant files are available; new GPU inference is not intrinsically required.

## Evidence release and citation work still outstanding

- Obtain and review the 224 result files, split definitions and the missing 16-bird great-tit/zebra-finch construction manifests. Release a permission-checked evidence bundle with checksums, excluding restricted media, restricted embeddings and disallowed identity annotations. The current public release does not support an end-to-end numerical reproduction.
- Recover the original model-specific package inventories, checkpoint revisions, weight digests and numerical-precision settings from the retained run records. Do not substitute today's package versions for the environment actually used. BirdNET `1.1.0` and Bacpipe `1.3.5` are documented; they are not a full GPU environment lock.
- Verify whether the BirdNET v3 preview weights match Zenodo record 18247420 before identifying that deposit as the exact model used.
- Archive the corrected software release on Zenodo when an authenticated publishing account is available; cite its version DOI. A software DOI is useful, but is not listed as a prerequisite in the [bioRxiv submission guide](https://www.biorxiv.org/submit-a-manuscript).
- After bioRxiv assigns the preprint DOI, update `CITATION.cff`, README and the research page. Do not invent or reserve a preprint DOI locally.

The Djuma/BirdNET-Cloud geography review, broader website presentation changes, journal-length editing and proposed additional GPU experiments are separate work.
