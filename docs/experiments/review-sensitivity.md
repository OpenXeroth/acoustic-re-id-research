# Review sensitivity analyses

Planned before running the new analyses, in response to methodological review.
These are follow-up analyses, not prospectively registered original experiments.

## Fixed design

- Published BirdNET and Perch vectors: chiffchaff, little owl and tree pipit.
  Use exactly the recordists' `train`/`test` clip pool, excluding tree pipit's
  additional `more`/`years` entries from both arms. Preserve every individual's
  enrolment and query count. Randomise clips within individual at seeds 101–110.
  Keep the original three heads, enrolment-only preprocessing and ridge lambda 1.
  Report the mean and range across seeds, ordinary and balanced accuracy,
  paired individual-bootstrap intervals (10,000 draws, seed 17) on the mean
  difference, and every seed's predictions and split hash. Intervals condition
  on these ten splits; seed ranges are reported separately. A positive result
  does not isolate duplication, background, vocal drift or session coverage.
- For each split, report raw unit-vector similarity to the nearest enrolment
  clip of the same individual. Where only a published train/test block is
  available, call it a block, not a verified recording session. Similarity is a
  descriptive diagnostic, not proof of near-duplication or its causal effect.
- Open set: retain the original 16 role allocations, frozen selected vectors,
  budgets and threshold rule. Compare mean-profile and nearest-enrolment-clip
  scoring; each scorer calibrates its own thresholds on calibration strangers.
  Never choose a scorer or threshold using the test results. Report both.
- Great tit 50-bird calibration sensitivity: retain the original galleries and
  test roles. Use nested subsets of 4, 8 and 12 calibration strangers, selected
  by sorted identity within the existing independently allocated calibration
  role. Do not change test animals or claim monotonic improvement.
- BirdNET paired-clip diagnostic: use the original query clips of the two
  BirdPark groups and the across-year rook. Compare same-individual/different-
  recording pairs with different-individual/same-recording pairs. Report
  same-day different-recording pairs separately if they exist, plus 1–7-day
  and longer gaps. Do not infer absent dates or silently drop an endpoint.
- Report the existing background/call balanced accuracies and separate
  majority-class baselines. No new model inference is needed.

## Boundaries and provenance

Results do not exist until the runs finish. Run on xen1 using existing local
vectors and data. Retain source commit and digest, this protocol's hash, lockfile
hash, runtime package inventory, input hashes, seeds, predictions and splits.
Set `XINYENYANA_RESULT_ARCHIVE=gs://xinyenyana/results`; archive every output by
its byte hash and verify it. Publish only aggregate projections, never animal
identities, media, private annotations or embeddings. Original results remain
unchanged. Donor rotation is deferred; state the single fixed-donor limitation.
