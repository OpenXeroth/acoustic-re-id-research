# Public measurement protocol

This is a retrospective guide to the released implementation. The prospective records were kept in the authors' internal repository; this page is not a reconstructed preregistration or a replacement for that history.

The current method is described in Appendix S1 of the [supporting information](supporting-information.md). It covers dataset splits, enrolment-only layer selection, the fixed kernel-ridge classifier, recording/background controls, within-recording shuffles and independently calibrated open-set thresholds. Random clip splits are diagnostic comparisons to session splits, not a recommended validation design. Djuma audio is not benchmark input.

Reported analyses use 9,999 label shuffles and 10,000 individual bootstrap draws where stated. Some CLI defaults are smaller; specify the reported counts explicitly. `scripts/paper_v6_evidence.py` checks role-specific contracts, with the count-check exceptions stated in S1.14. Its historical archive contracts remain distinct from the v8 correction scope.

For cosine score normalisation and the main open-set sweep, exclude the raw duration and loudness controls: normalising to unit length discards their magnitude. Their per-dimension-standardised kernel-ridge comparisons remain valid. For paired verification, tied scores must cross a threshold together. See [publication status](publication-status.md) for the pending numerical impact check.

Every independent run must retain its configuration, input and split hashes, source-code digest, environment and weight versions, seeds, predictions and metrics. See [reproduction](reproduction.md) for local archiving and the public aggregate evidence deposit and its limits.
