# Historical A3 call-cleaning comparison

`src/xinyenyana/a3.py` implements five configurations: no cleaning; trimming; band filtering; spectral denoising; and all three together. The no-cleaning arm is rerun in the same experiment so comparisons use the same code and environment. The endpoint frequency band is learned from enrolment clips only.

Trimming is not evaluated on the motor layer, whose note-gap timing features would otherwise measure the trimming operation. Empty-note and missing-noise-estimate outcomes are retained in the result diagnostics.

This is documentation for a historical code path, not a newly run experiment or a claim about the paper's remedies. Those remedies and their limitations are described in Appendix S1.11 of the [supporting information](supporting-information.md). The original internal protocol/history is not bundled; see [measurement protocol](measurement-protocol.md).
