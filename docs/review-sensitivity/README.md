# Appendix S2. Follow-up sensitivity analyses after review

These analyses were specified after review, before their execution, and are not the original preregistered analyses. The fixed protocol was committed as 84c26f1664d95eea36923d7b826bf1a0ebf7f433; its SHA-256 is 22d2276709936df17b48ef208e5211dff7b3c69838afe4d3d9856242f927b0ec. The open-set runner was corrected to retain the original rook enrolment cap before any rook result was written (commit 7d52438). All 194 new outputs were copied to the content-addressed GCS archive with checksum verification. Public aggregate projections, source hashes, split hashes and runtime inventories are included in the v9 evidence deposit (https://doi.org/10.5281/zenodo.23167851). Software 0.3.0 is archived at https://doi.org/10.5281/zenodo.23167850. Recordings, embeddings, real identities and per-clip observations are not redistributed.

## S2.1 Count-matched clip splits

Randomisation describes how units are assigned; it does not define the unit. Randomly assigning whole recording sessions can be consistent with a valid session-disjoint evaluation. Randomly assigning individual clips can instead mix the context of a session between enrolment and test. Which design is appropriate depends on the target use. Recognition of a familiar animal in a later session requires evaluation on sessions untouched by training and model selection.

For each species and encoder, both arms used exactly the published train/test clip pool: chiffchaff 5,107/1,131 enrolment/test clips, little owl 545/407 and tree pipit 409/303. The tree-pipit more/years entries were excluded in both arms. Each bird retained its own enrolment/test counts. Ten fixed seeds (101–110) shuffled clips within bird. Both arms used the same frozen vectors, enrolment-only preprocessing, three classifier rules and ridge penalty 1.0. The paired bootstrap resampled individuals jointly over the session arm and mean of the ten random arms (10,000 draws, seed 17); its intervals condition on those ten fixed randomisations. They are not intervals over all possible split choices. All 18 mean differences were positive, from 0.196 to 0.508, with every interval excluding zero.

Detailed results: values below give session accuracy; mean random-clip accuracy [minimum–maximum across ten seeds]; mean difference [paired 95% interval]; and balanced accuracy, session → random mean. Balanced accuracy is mean recall over birds.

littleowl, birdnet, kernel ridge: 0.477; 0.963 [0.953–0.973]; 0.486 [0.359–0.606]; balanced 0.489 → 0.961.

littleowl, birdnet, class mean: 0.430; 0.897 [0.880–0.912]; 0.467 [0.343–0.582]; balanced 0.452 → 0.894.

littleowl, birdnet, nearest clip: 0.577; 0.973 [0.961–0.980]; 0.396 [0.249–0.546]; balanced 0.570 → 0.973.

littleowl, google-perch, kernel ridge: 0.631; 0.964 [0.958–0.973]; 0.333 [0.201–0.479]; balanced 0.625 → 0.964.

littleowl, google-perch, class mean: 0.543; 0.879 [0.865–0.899]; 0.336 [0.203–0.466]; balanced 0.560 → 0.876.

littleowl, google-perch, nearest clip: 0.582; 0.963 [0.961–0.971]; 0.381 [0.229–0.537]; balanced 0.591 → 0.964.

chiffchaff, birdnet, kernel ridge: 0.734; 0.969 [0.963–0.975]; 0.235 [0.147–0.315]; balanced 0.742 → 0.947.

chiffchaff, birdnet, class mean: 0.502; 0.810 [0.792–0.827]; 0.308 [0.219–0.425]; balanced 0.515 → 0.796.

chiffchaff, birdnet, nearest clip: 0.595; 0.941 [0.930–0.953]; 0.346 [0.253–0.434]; balanced 0.602 → 0.916.

chiffchaff, google-perch, kernel ridge: 0.751; 0.947 [0.935–0.957]; 0.196 [0.120–0.270]; balanced 0.735 → 0.909.

chiffchaff, google-perch, class mean: 0.404; 0.622 [0.599–0.653]; 0.218 [0.128–0.276]; balanced 0.456 → 0.626.

chiffchaff, google-perch, nearest clip: 0.486; 0.818 [0.809–0.842]; 0.332 [0.225–0.389]; balanced 0.494 → 0.785.

pipit, birdnet, kernel ridge: 0.419; 0.927 [0.904–0.947]; 0.508 [0.328–0.688]; balanced 0.404 → 0.922.

pipit, birdnet, class mean: 0.363; 0.782 [0.746–0.815]; 0.418 [0.291–0.548]; balanced 0.342 → 0.772.

pipit, birdnet, nearest clip: 0.406; 0.865 [0.835–0.908]; 0.459 [0.325–0.592]; balanced 0.395 → 0.863.

pipit, google-perch, kernel ridge: 0.452; 0.910 [0.888–0.944]; 0.458 [0.275–0.635]; balanced 0.445 → 0.906.

pipit, google-perch, class mean: 0.277; 0.717 [0.653–0.772]; 0.440 [0.339–0.543]; balanced 0.270 → 0.709.

pipit, google-perch, nearest clip: 0.356; 0.771 [0.749–0.809]; 0.415 [0.263–0.569]; balanced 0.356 → 0.767.

The median cosine similarity to the nearest enrolment clip from the same bird was higher in every randomisation for each of the six species–encoder combinations. This is a descriptive measure on raw unit-normalised vectors, not proof of near-duplicate calls or a causal estimate of the contribution of background, vocal drift or recording conditions. The retained metadata identify original train/test blocks, not finer recording sessions; we do not infer missing session labels.

## S2.2 Original published-fold comparisons retained

The following are the original Table 3 comparisons, preserved for transparency. Unlike S2.1, the published little-owl fold used 761 rather than 545 enrolment clips, and the tree-pipit fold used additional clips. They therefore do not isolate split composition from training size or clip pool. For each row: original random-fold accuracy; session accuracy; original paired difference [95% interval]. The original rook comparison used one count-matched random split, not the ten-seed procedure.

Little owl, BirdNET, kernel ridge: 0.984; 0.477; 0.508 (0.378 to 0.626).

Little owl, BirdNET, class mean: 0.927; 0.430; 0.497 (0.365 to 0.609).

Little owl, BirdNET, nearest clip: 0.979; 0.577; 0.402 (0.265 to 0.542).

Little owl, Perch, kernel ridge: 0.990; 0.631; 0.358 (0.222 to 0.507).

Little owl, Perch, class mean: 0.880; 0.543; 0.337 (0.196 to 0.474).

Little owl, Perch, nearest clip: 0.979; 0.582; 0.397 (0.249 to 0.548).

Chiffchaff, BirdNET, kernel ridge: 0.961; 0.734; 0.227 (0.149 to 0.309).

Chiffchaff, BirdNET, class mean: 0.800; 0.502; 0.298 (0.219 to 0.444).

Chiffchaff, BirdNET, nearest clip: 0.935; 0.595; 0.340 (0.253 to 0.429).

Chiffchaff, Perch, kernel ridge: 0.944; 0.751; 0.193 (0.125 to 0.290).

Chiffchaff, Perch, class mean: 0.591; 0.404; 0.187 (0.099 to 0.287).

Chiffchaff, Perch, nearest clip: 0.819; 0.486; 0.333 (0.243 to 0.403).

Tree pipit (not the same clips), BirdNET, kernel ridge: 0.875; 0.419; 0.456 (0.270 to 0.645).

Tree pipit (not the same clips), BirdNET, class mean: 0.670; 0.363; 0.307 (0.145 to 0.463).

Tree pipit (not the same clips), BirdNET, nearest clip: 0.839; 0.406; 0.433 (0.307 to 0.558).

Tree pipit (not the same clips), Perch, kernel ridge: 0.894; 0.452; 0.442 (0.249 to 0.629).

Tree pipit (not the same clips), Perch, class mean: 0.575; 0.277; 0.298 (0.212 to 0.403).

Tree pipit (not the same clips), Perch, nearest clip: 0.700; 0.356; 0.343 (0.198 to 0.484).

Rook, shared aviary, BirdNET (this study), kernel ridge: 0.547; 0.422; 0.126 (0.065 to 0.172).

## S2.3 Background accuracy and class imbalance

These are the retained BirdNET kernel-ridge background/call results, not new model inference. Balanced accuracy gives equal weight to each bird. The query-majority reference is the fraction of clips belonging to the commonest test identity; it is a descriptive best-constant reference using test labels. The separate fitted constant baseline always predicts the most frequent enrolment identity. Ties are resolved by the fixed anonymous identity ordering. These two quantities answer different questions.

chiffchaff-withinyear: foreground: ordinary 0.765, balanced 0.791, query-majority 0.393, enrolment-majority predictor 0.062 (n=1131); background: ordinary 0.737, balanced 0.714, query-majority 0.395, enrolment-majority predictor 0.063 (n=1100).

chiffchaff-acrossyear: foreground: ordinary 0.160, balanced 0.151, query-majority 0.160, enrolment-majority predictor 0.100 (n=200); background: ordinary 0.132, balanced 0.143, query-majority 0.152, enrolment-majority predictor 0.096 (n=197).

littleowl-acrossyear: foreground: ordinary 0.501, balanced 0.518, query-majority 0.084, enrolment-majority predictor 0.044 (n=407); background: ordinary 0.105, balanced 0.108, query-majority 0.086, enrolment-majority predictor 0.044 (n=409).

pipit-withinyear: foreground: ordinary 0.376, balanced 0.362, query-majority 0.125, enrolment-majority predictor 0.116 (n=303); background: ordinary 0.177, balanced 0.177, query-majority 0.126, enrolment-majority predictor 0.116 (n=293).

pipit-acrossyear: foreground: ordinary 0.214, balanced 0.218, query-majority 0.160, enrolment-majority predictor 0.073 (n=313); background: ordinary 0.065, balanced 0.056, query-majority 0.160, enrolment-majority predictor 0.072 (n=306).

Within-year chiffchaff background identification remains substantial under balanced accuracy (0.714 against 0.791 for calls). The two balanced accuracies should not be described as equivalent. Calls and backgrounds have slightly different test counts and therefore separate majority references.

## S2.4 Open-set scoring and calibration-size sensitivity

The original mean-profile scorer was compared with a nearest-enrolment-clip scorer, using the same frozen selected embeddings and all 16 original allocations. Each scorer independently calibrated its own threshold using only the calibration strangers. The nearest-clip scorer assigns the identity of the most similar enrolled clip; no test clip enters its gallery. Model selection, preprocessing, role assignments and stranger budgets were unchanged. The original 90-clip-per-bird rook enrolment cap was retained. Recomputed mean-profile closed-set accuracy and 10% budget known-correct acceptance reproduced the archived values in every allocation. Both scorers are reported; the better test result is not a deployment-selected model.

Across the 36 distinct neural networks, the highest median accepted-and-correctly-named proportion at a 10% calibration stranger budget was 0.374 with mean profiles and 0.462 with nearest-clip scoring on the four wild session-disjoint endpoints. Both maxima were Perch 2.0 on little owl. For that nearest-clip comparison, the range over allocations was 0.000–0.596, and the actual test-stranger acceptance rate had median 0.099 and range 0.000–0.615. This improvement is conditional on the tested models, scorers, roles and data; neither number is a ceiling on future re-identification methods. Nearest-clip scoring did not improve every model and dataset: BirdNET on rook fell from 0.080 to 0.037.

Detailed comparisons for BirdNET and Perch 2.0: mean-profile → nearest-clip known-correct acceptance, followed by the actual stranger acceptance rates for the same two scorers. All values are medians [allocation minima–maxima], using the original 16 allocations and 10% calibration budget. Table S15 retains all original mean-profile results; the complete two-scorer aggregates for all 36 distinct networks are in the v9 evidence and public repository. The numerically identical WavLM Base+ speaker-verification duplicate is retained in the archive but excluded from the network count.

Great tit, 16 birds, birdnet-v2.4: known 0.175 [0.050–0.350] → 0.250 [0.025–0.400]; strangers 0.087 [0.000–0.350] → 0.087 [0.000–0.250].

Great tit, 16 birds, perch-v2: known 0.225 [0.025–0.350] → 0.300 [0.175–0.500]; strangers 0.087 [0.000–0.275] → 0.100 [0.000–0.300].

Great tit, 50 birds, birdnet-v2.4: known 0.092 [0.015–0.177] → 0.144 [0.050–0.267]; strangers 0.092 [0.025–0.275] → 0.107 [0.017–0.269].

Great tit, 50 birds, perch-v2: known 0.112 [0.015–0.250] → 0.216 [0.085–0.292]; strangers 0.103 [0.025–0.277] → 0.096 [0.017–0.246].

Little owl, birdnet-v2.4: known 0.200 [0.000–0.520] → 0.229 [0.000–0.650]; strangers 0.084 [0.000–0.697] → 0.074 [0.000–0.679].

Little owl, perch-v2: known 0.374 [0.000–0.500] → 0.462 [0.000–0.596]; strangers 0.100 [0.000–0.587] → 0.099 [0.000–0.615].

Rook, birdnet-v2.4: known 0.080 [0.000–0.234] → 0.037 [0.002–0.172]; strangers 0.100 [0.000–0.503] → 0.099 [0.009–0.277].

Rook, perch-v2: known 0.101 [0.000–0.291] → 0.133 [0.046–0.264]; strangers 0.100 [0.000–0.403] → 0.101 [0.026–0.229].

Cockatoo, published random fold, birdnet-v2.4: known 0.741 [0.029–0.921] → 0.863 [0.000–1.000]; strangers 0.082 [0.000–0.584] → 0.086 [0.000–0.636].

Cockatoo, published random fold, perch-v2: known 0.814 [0.029–0.941] → 0.883 [0.088–1.000]; strangers 0.085 [0.000–0.632] → 0.077 [0.000–0.781].

Calibration size was assessed on the 50-bird great-tit endpoint using nested subsets of 4, 8 and 12 calibration strangers selected by fixed sorted identity order within the original calibration role. Galleries, calibration-known birds and both test roles stayed unchanged. The original allocation sometimes had 13 calibration strangers; the 12-stranger variant is therefore not always identical to the original. This is a controlled sensitivity check conditional on these subsets, not a general learning curve over randomly sampled calibration sets.

birdnet-v2.4, mean-profile, 4 calibration strangers: known 0.075 [0.000–0.158]; strangers 0.079 [0.008–0.275]; 5/16 allocations exceeded the 10% test-stranger budget.

birdnet-v2.4, mean-profile, 8 calibration strangers: known 0.071 [0.015–0.175]; strangers 0.075 [0.008–0.333]; 5/16 allocations exceeded the 10% test-stranger budget.

birdnet-v2.4, mean-profile, 12 calibration strangers: known 0.084 [0.015–0.154]; strangers 0.092 [0.025–0.250]; 7/16 allocations exceeded the 10% test-stranger budget.

birdnet-v2.4, nearest-clip, 4 calibration strangers: known 0.136 [0.023–0.308]; strangers 0.096 [0.000–0.269]; 7/16 allocations exceeded the 10% test-stranger budget.

birdnet-v2.4, nearest-clip, 8 calibration strangers: known 0.128 [0.038–0.308]; strangers 0.067 [0.000–0.269]; 5/16 allocations exceeded the 10% test-stranger budget.

birdnet-v2.4, nearest-clip, 12 calibration strangers: known 0.142 [0.050–0.267]; strangers 0.103 [0.017–0.269]; 8/16 allocations exceeded the 10% test-stranger budget.

perch-v2, mean-profile, 4 calibration strangers: known 0.109 [0.015–0.250]; strangers 0.068 [0.000–0.308]; 5/16 allocations exceeded the 10% test-stranger budget.

perch-v2, mean-profile, 8 calibration strangers: known 0.115 [0.015–0.250]; strangers 0.107 [0.000–0.283]; 8/16 allocations exceeded the 10% test-stranger budget.

perch-v2, mean-profile, 12 calibration strangers: known 0.112 [0.023–0.250]; strangers 0.103 [0.025–0.277]; 8/16 allocations exceeded the 10% test-stranger budget.

perch-v2, nearest-clip, 4 calibration strangers: known 0.204 [0.108–0.383]; strangers 0.092 [0.000–0.369]; 7/16 allocations exceeded the 10% test-stranger budget.

perch-v2, nearest-clip, 8 calibration strangers: known 0.212 [0.108–0.308]; strangers 0.088 [0.017–0.215]; 7/16 allocations exceeded the 10% test-stranger budget.

perch-v2, nearest-clip, 12 calibration strangers: known 0.216 [0.085–0.292]; strangers 0.092 [0.017–0.246]; 6/16 allocations exceeded the 10% test-stranger budget.

Increasing calibration size did not yield a monotonic improvement. For BirdNET with mean profiles, median known-correct acceptance was 0.075, 0.071 and 0.084 with 4, 8 and 12 strangers; for Perch 2.0 with nearest-clip scoring it was 0.204, 0.212 and 0.216. Test-stranger rates still exceeded the calibration target in some allocations at every size. Small calibration sets are a plausible source of variability but this check does not show that enlarging them resolves transfer between strangers.

## S2.5 Timing of paired clips

Using retained BirdNET test embeddings, we enumerated all unordered same-individual/different-recording pairs and different-individual/same-recording pairs. Dates came from the original recording metadata or dated BirdPark recording names. We report cosine means for same-day, 1–7-day and longer gaps. Pairs share clips and individuals; these are descriptive aggregates, not independent biological replicates or additional significance tests. The original sampled-pair error rates are unchanged.

BirdPark group of four: animal 1 to 7 days: 0 pairs, mean not estimable; animal all: 751 pairs, mean 0.818; animal over 7 days: 751 pairs, mean 0.818; animal same day: 0 pairs, mean not estimable; recording: 3,953 pairs, mean 0.657.

BirdPark group of eight: animal 1 to 7 days: 0 pairs, mean not estimable; animal all: 418 pairs, mean 0.726; animal over 7 days: 418 pairs, mean 0.726; animal same day: 0 pairs, mean not estimable; recording: 6,577 pairs, mean 0.729.

Rook: animal 1 to 7 days: 205,017 pairs, mean 0.765; animal all: 1,168,249 pairs, mean 0.764; animal over 7 days: 719,210 pairs, mean 0.770; animal same day: 244,022 pairs, mean 0.748; recording: 56,625 pairs, mean 0.731.

All 751 and 418 eligible same-bird, different-recording pairs in the two BirdPark groups were more than seven days apart. A within-day or within-week BirdPark control therefore cannot be run with these test clips. The rook’s same-day animal mean (0.748) exceeded its recording-pair mean (0.731), but this does not match elapsed time exactly, remove other context or establish temporal invariance. The paired diagnostic can reflect both identity persistence and the interval between recordings; that qualification applies to biological interpretation of the original result.

## S2.6 Fixed donors, interpretation and access

The added-background experiment used one fixed sorted-order donor mapping. Its bootstrap varies recipients conditional on that mapping; it does not quantify variation over donors. Donor rotation remains an unperformed robustness experiment. No new claim of donor-independent generality is made.

For users seeking better models, choose a split that matches the intended deployment, retain untouched sessions for final evaluation, and tune architecture or training diversity only on development data. A random-clip score measures performance under that clip-sharing design; raising it does not by itself demonstrate improved recognition in a new session. In open-set work, report identification, correct acceptance and actual stranger acceptance together, and calibrate each scorer separately without selecting on test outcomes.

Full aggregates and reproducible code: https://github.com/OpenXeroth/acoustic-re-id-research/tree/main/docs/review-sensitivity. The manuscript, supporting information, figures and original evidence documentation are licensed CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/); software remains Apache-2.0. Original datasets and model weights retain their publishers’ terms.

[Fixed protocol](../experiments/review-sensitivity.md) · [Result manifest](results/manifest.json) · [Split summary](split-and-time.json) · [Background summary](background.json)
