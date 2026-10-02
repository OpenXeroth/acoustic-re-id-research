"""The paper's main and supporting tables, in the manuscript's wording, from the built ledgers.

Run after scripts/paper_v6_build.sh. Every value comes from a ledger written by a gated
renderer in the build directory, or from a result file whose SHA-256 the evidence report
lists. Numbers are rounded here for display only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from paper_v6_controls import PAIRINGS, complete_gate
from paper_v6_evidence import BACKGROUND_ENDPOINTS, ENDPOINTS
from paper_v6_overview import DISPLAY

NAMES = {
    "birdpark-juv01": "Zebra finch, group of four",
    "birdpark-juv03": "Zebra finch, group of eight",
    "rookid-full-width": "Rook, across year",
    "zebra-finch": "Zebra finch, one bird per recording",
    "chiffchaff-withinyear": "Chiffchaff, within year",
    "chiffchaff-acrossyear": "Chiffchaff, across year",
    "littleowl-acrossyear": "Little owl, across year",
    "pipit-withinyear": "Tree pipit, within year",
    "pipit-acrossyear": "Tree pipit, across year",
    "great-tit": "Great tit, across year",
    "penguin-acrossnight": "Little penguin, across night",
    "cockatoo-fold1": "Red-tailed black cockatoo, random split",
    "bat-acrosstreatment": "Egyptian fruit bat, across year",
    "great-tit-full": "Great tit, 50 birds, across year",
}
ORDER = [e for e in NAMES if e != "great-tit-full"]
TABLE1_NAMES = {
    "rookid-full-width": "Rook",
    "great-tit": "Great tit",
    "penguin-acrossnight": "Little penguin",
    "cockatoo-fold1": "Red-tailed black cockatoo",
    "bat-acrosstreatment": "Egyptian fruit bat",
}
HEADLINE_STEM = {e: e for e in ORDER} | {"rookid-full-width": "rook-full-across-year-capped-90"}
SPLIT_BY = {
    "birdpark-juv01": "recording file",
    "birdpark-juv03": "recording file",
    "rookid-full-width": "year",
    "zebra-finch": "recording",
    "chiffchaff-withinyear": "recording day",
    "chiffchaff-acrossyear": "year",
    "littleowl-acrossyear": "year",
    "pipit-withinyear": "session within year",
    "pipit-acrossyear": "year",
    "great-tit": "year and nest box",
    "penguin-acrossnight": "night",
    "cockatoo-fold1": "random over clips",
    "bat-acrosstreatment": "year, day as session",
}
#: Table 1 columns that describe the published data, from the sources' own descriptions.
DATASET = {
    "birdpark-juv01": (
        "*Taeniopygia guttata*",
        "Rüttimann et al. (2024)",
        "shared microphone, captive",
        "body-worn transmitter",
        "no",
    ),
    "birdpark-juv03": (
        "*T. guttata*",
        "Rüttimann et al. (2024)",
        "shared microphone, captive",
        "body-worn transmitter",
        "no",
    ),
    "rookid-full-width": (
        "*Corvus frugilegus*",
        "Martin et al. (2022a, b)",
        "shared microphones, captive aviary",
        "leg rings, observer",
        "no",
    ),
    "zebra-finch": (
        "*T. guttata*",
        "Elie & Theunissen (2016, 2018)",
        "one bird per recording, captive",
        "recordist",
        "no",
    ),
    "chiffchaff-withinyear": (
        "*Phylloscopus collybita*",
        "Stowell et al. (2018, 2019)",
        "own territory, wild",
        "recordist, territory",
        "yes",
    ),
    "chiffchaff-acrossyear": (
        "*P. collybita*",
        "Stowell et al. (2018, 2019)",
        "own territory, wild",
        "recordist, territory",
        "yes",
    ),
    "littleowl-acrossyear": (
        "*Athene noctua*",
        "Stowell et al. (2018, 2019)",
        "own territory, wild",
        "recordist, territory",
        "yes",
    ),
    "pipit-withinyear": (
        "*Anthus trivialis*",
        "Stowell et al. (2018, 2019)",
        "own territory, wild",
        "recordist, territory",
        "yes",
    ),
    "pipit-acrossyear": (
        "*A. trivialis*",
        "Stowell et al. (2018, 2019)",
        "own territory, wild",
        "recordist, territory",
        "yes",
    ),
    "great-tit": (
        "*Parus major*",
        "Merino Recalde et al. (2024)",
        "nest box, wild",
        "ringed bird at nest box",
        "no",
    ),
    "penguin-acrossnight": (
        "*Eudyptula minor*",
        "Huang et al. (2025a, b)",
        "nest, wild",
        "nest",
        "no",
    ),
    "cockatoo-fold1": (
        "*Calyptorhynchus banksii*",
        "Huang et al. (2025a, b)",
        "nest, wild",
        "nest",
        "no",
    ),
    "bat-acrosstreatment": (
        "*Rousettus aegyptiacus*",
        "Prat et al. (2017a)",
        "one bat per recording, captive",
        "synchronised video",
        "no",
    ),
}
#: The recording variable Table 1 reports NMI for, as (confound axis, word used).
NMI_AXES = {
    "birdpark-juv01": [("session", "session")],
    "birdpark-juv03": [("session", "session")],
    "rookid-full-width": [("recording", "recording")],
    "zebra-finch": [("recording_day", "day")],
    "great-tit": [("source_recording", "recording")],
    "penguin-acrossnight": [("recording_day", "day")],
    "bat-acrosstreatment": [("source_recording", "recording"), ("recording_day", "day")],
}
AXIS_WORDS = {
    "source_recording": "recording",
    "recording": "recording",
    "session": "session",
    "date": "date",
    "recording_day": "day",
    "recorded": "recording time",
    "channel": "microphone channel",
    "treatment": "treatment",
    "folder": "folder",
    "nestbox": "nest box",
    "cohort": "cohort",
    "year": "year",
    "call_type": "call type",
    "span": "session span",
    "event": "calling event",
    "source_channels": "channels stored",
    "microphone": "microphone",
}
HEAD = {
    "kernel ridge to one-hot": "kernel ridge",
    "mean cosine similarity to each identity": "class mean",
    "single nearest enrollment clip, cosine": "nearest clip",
}
LEVELS = [("-10db", "+10 dB"), ("0db", "0 dB"), ("10db", "−10 dB")]
PAIR_WORDS = {
    "foreground_gallery_foreground_query": "Fitted on calls, tested on calls",
    "background_gallery_background_query": "Fitted on backgrounds, tested on backgrounds",
    "foreground_gallery_background_query": "Fitted on calls, tested on backgrounds",
    "background_gallery_foreground_query": "Fitted on backgrounds, tested on calls",
}
SHORT_PAIR = {
    "foreground_gallery_foreground_query": "C→C",
    "background_gallery_background_query": "B→B",
    "foreground_gallery_background_query": "C→B",
    "background_gallery_foreground_query": "B→C",
}
#: The speaker-verification WavLM Base+ checkpoint has the weights and every value of
#: WavLM Base+, so it is not counted as a separate entry.
DUPLICATE = "wavlm-base-plus-sv"
MODELS = [m for m in DISPLAY if m != DUPLICATE]


def f3(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}".replace("-", "−")


def bound(value: float) -> str:
    """An interval bound; one that is not zero but rounds to zero keeps its first digit."""
    if value != 0 and round(value, 3) == 0:
        digits = 4
        while round(value, digits) == 0:
            digits += 1
        return f"{value:.{digits}f}".replace("-", "−")
    return f3(value)


def iv(value: float, bounds: list[float]) -> str:
    return f"{f3(value)} ({bound(bounds[0])} to {bound(bounds[1])})"


def p4(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def p_draws(value: float | None, bound: float) -> str:
    """A Holm-adjusted p from the share of 10,000 draws on each side of zero.

    When no draw crossed zero the unadjusted two-sided p is below 2 / 10,000, so the adjusted
    value is below the family size times that; ``bound`` is that limit.
    """
    if value is None:
        return "n/a"
    return f"< {bound:.4f}" if value == 0 else f"{value:.4f}"


def pairing_cell(row: dict[str, Any], prefix: str) -> str:
    return f"{iv(row['accuracy'], row['interval_95'])}; {prefix}{p4(row['permutation_p'])}"


def span(row: dict[str, float]) -> str:
    return f"{f3(row['median'])} ({f3(row['minimum'])} to {f3(row['maximum'])})"


def table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def short(model: str) -> str:
    return model.split("/")[-1]


class Build:
    def __init__(self, build: Path, results: Path) -> None:
        self.build, self.results = build, results
        self.evidence = complete_gate(build / "evidence.json")
        self.review = self.ledger("review/review-numbers.json")

    def ledger(self, relative: str) -> dict[str, Any]:
        data = json.loads((self.build / relative).read_text())
        return {k: v["value"] for k, v in data["values"].items()}

    def result(self, name: str) -> dict[str, Any]:
        raw = (self.results / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.evidence["sources"][name]:
            raise ValueError(f"{name}: bytes differ from the evidence report")
        return json.loads(raw)


def individuals_cell(enrolled: int, tested: int) -> str:
    return str(enrolled) if tested == enrolled else f"{enrolled} ({tested} in test)"


def main_tables(b: Build) -> str:
    headline = b.ledger("headline/headline-numbers-v6.json")
    carried = b.ledger("carried/numbers.json")
    controls = b.ledger("recording-controls/recording-controls-numbers.json")
    opened = b.ledger("open-set/open-set-numbers.json")
    confound = carried["confound_index"]
    out = []

    tested = {
        ep: entry["Fisher"]["paired"]["animals"]
        for ep, entry in b.review["comparison"]["per_endpoint"].items()
    }
    rows = []
    for ep in ORDER:
        h = headline[f"headline.{HEADLINE_STEM[ep]}"]
        species, source, design, attributed, backgrounds = DATASET[ep]
        nmi = (
            "; ".join(
                f"{f3(confound[ep]['axes'][axis]['all']['nmi'])} ({word})"
                for axis, word in NMI_AXES.get(ep, [])
            )
            or "not recorded"
        )
        rows.append(
            [
                TABLE1_NAMES.get(ep, NAMES[ep]),
                species,
                source,
                individuals_cell(h["individuals"], tested[ep]),
                f"{h['enrolment_clips']:,} / {h['query_clips']:,}",
                design,
                attributed,
                SPLIT_BY[ep],
                backgrounds,
                nmi,
            ]
        )
    out.append(
        "**Table 1.** The 13 datasets used in this study.\n\n"
        + table(
            [
                "Dataset",
                "Species",
                "Source",
                "n",
                "Enrolment / test clips",
                "Recording design",
                "Identity attributed by",
                "Split by",
                "Background recordings",
                "NMI, identity and recording",
            ],
            rows,
        )
        + "\n\n*Notes:* n, number of individuals enrolled; where fewer had test clips, "
        "that number follows in brackets. Recording design: whether each animal was "
        "recorded at its own site or several animals shared recordings. Identity attributed "
        "by: how the original authors established which animal produced each call; in most "
        "designs this also fixes where each animal was recorded. Split by: the recording "
        "session that separates enrolment from test clips. Background recordings: whether "
        "ambient recordings without the animal are available. NMI: mutual information between "
        "identity and the recording variable named, over all clips, divided by the smaller of "
        "the two entropies (0, unrelated; 1, either variable determines the other, as when "
        "every recording holds one animal); not recorded, the dataset does not identify "
        "recordings. The rook row is the dataset with 11 birds; the three chiffchaff and "
        "tree pipit rows and the little owl row are separate datasets from one study."
    )

    rows = []
    for ep in ORDER:
        h = headline[f"headline.{HEADLINE_STEM[ep]}"]
        within = (
            "n/a"
            if h["within_95"] is None or ep in NO_SHARED_RECORDINGS
            else f"{f3(h['within_95'])} ({p4(h['within_p'])})"
        )
        three = (
            "n/a"
            if h["three_stage"] is None
            else f"{f3(h['three_stage'][0])} to {f3(h['three_stage'][1])}"
        )
        rows.append(
            [
                NAMES[ep],
                h["individuals"],
                SPLIT_BY[ep],
                iv(h["accuracy"], h["interval"]),
                three,
                f3(h["majority"]),
                f3(h["permutation_95"]),
                p4(h["permutation_p"]),
                within,
            ]
        )
    out.append(
        "**Table 2.** Closed-set re-ID accuracy with BirdNET and the kernel ridge "
        "classifier.\n\n"
        + table(
            [
                "Dataset",
                "n",
                "Split by",
                "Re-ID accuracy (95% interval)",
                "Three-stage interval",
                "Majority class",
                "Permutation 95th percentile",
                "p",
                "Within-recording permutation 95th percentile (p)",
            ],
            rows,
        )
        + "\n\n*Notes:* The 95% interval resamples individuals 10,000 times; the three-stage "
        "interval also resamples recordings within individuals and clips within recordings, "
        "where the dataset records them. Majority class: accuracy of always naming the "
        "individual with the most test clips. Permutation: 95th percentile and p value from "
        "9,999 shuffles of the enrolment labels (0.0001 is the smallest attainable p). "
        "Within-recording permutation: labels shuffled only among clips of the same recording "
        "(for the zebra finch recorded singly, the same day). n/a: the dataset records no "
        "recordings, or too few hold two individuals for the test to change the labels (fruit "
        "bat: 16 of 6,252 enrolment clips could be shuffled). The rook enrolment was capped at "
        "90 clips per bird."
    )

    rows = []
    species = {
        "littleowl": "Little owl",
        "chiffchaff": "Chiffchaff",
        "pipit": "Tree pipit (not the same clips)",
    }
    for r in carried["published_split"]:
        rows.append(
            [
                species[r["species"]],
                r["vectors"],
                HEAD[r["head"]],
                f3(r["random_fold"]),
                f3(r["recordists"]),
                iv(r["fall"], r["fall_interval"]),
            ]
        )
    rook = carried["rook_split"]
    rows.append(
        [
            "Rook, shared aviary",
            "BirdNET (this study)",
            "kernel ridge",
            f3(rook["random_fold"]),
            f3(rook["session"]),
            iv(rook["fall"], rook["fall_interval"]),
        ]
    )
    out.append(
        "**Table 3.** Effect of the data split on re-ID accuracy, with embeddings and "
        "classifier unchanged.\n\n"
        + table(
            [
                "Species",
                "Embeddings",
                "Classifier",
                "Random split",
                "Session split",
                "Difference (95% interval)",
            ],
            rows,
        )
        + "\n\n*Notes:* Random split: fold 1 of the published dataset, drawn over clips (for the "
        "rook, a random division within each bird that keeps its number of test clips). "
        "Session split: the recordists' division (for the rook, enrolment in 2020 and testing "
        "in 2021, without the enrolment cap of Table 2). Difference: random minus session, "
        "with an interval that applies the same 10,000 draws of individuals to both splits. "
        "Embeddings for the little owl, chiffchaff and tree pipit are those published by "
        "Huang et al. (2025b). Classifiers: kernel ridge; class mean, mean cosine similarity "
        "to each individual's enrolment clips; nearest clip, the most similar enrolment clip."
    )

    rows = []
    for ep in BACKGROUND_ORDER:
        h = headline[f"headline.{HEADLINE_STEM[ep]}"]
        block = controls[f"{ep}.controls.birdnet-v2.4"]["four_pairings"]
        rows.append(
            [NAMES[ep], h["individuals"], f3(h["uniform_chance"])]
            + [pairing_cell(block[k], "p ") for k in PAIRINGS]
        )
    out.append(
        "**Table 4.** Re-ID from calls and from ambient background recordings, with "
        "BirdNET.\n\n"
        + table(["Dataset", "n", "Uniform chance"] + [PAIR_WORDS[k] for k in PAIRINGS], rows)
        + "\n\n*Notes:* Re-ID accuracy (95% interval over individuals, 10,000 draws) and "
        "permutation p (9,999 shuffles) for a classifier fitted on one kind of clip and tested "
        "on the same or the other kind, using the recordists' session splits. All 38 "
        "embeddings and controls are in Table S5."
    )

    rows = []
    for ep in BACKGROUND_ORDER:
        c = controls[f"{ep}.controls.birdnet-v2.4"]["challenge"]
        for key, word in LEVELS:
            d = c["own_vs_other"][key]
            rows.append(
                [
                    NAMES[ep],
                    word,
                    f3(c["arms"]["own_" + key]["accuracy"]),
                    f3(c["arms"]["other_" + key]["accuracy"]),
                    f"{100 * d['answer_changed_fraction']:.1f}%",
                    iv(d["own_minus_other_macro_recall"], d["identity_block_difference_95"]),
                ]
            )
    out.append(
        "**Table 5.** Effect of adding a background recording to test clips, with "
        "BirdNET.\n\n"
        + table(
            [
                "Dataset",
                "Added background relative to test clip",
                "Own donor",
                "Other donor",
                "Answers changed",
                "Own minus other, balanced accuracy (95% interval)",
            ],
            rows,
        )
        + "\n\n*Notes:* The classifier is fitted on the original enrolment clips. Each test clip "
        "is mixed with a background from its own individual (own donor) or from another "
        "individual (other donor), 10 dB louder than the clip, at equal level or 10 dB "
        "quieter. Own and other donor: re-ID accuracy. Answers changed: proportion of test "
        "clips assigned differently in the two mixtures. Last column: paired difference in "
        "balanced accuracy (mean per-individual accuracy), interval over individuals (10,000 "
        "draws). All 38 embeddings and controls are in Table S6."
    )

    rows = []
    v5 = carried["open_set"]["great-tit"]["representations"]
    review_open = b.review["open_set"]
    for ep in (
        "great-tit",
        "great-tit-full",
        "littleowl-acrossyear",
        "rookid-full-width",
        "cockatoo-fold1",
    ):
        entry = opened[ep]
        r = review_open[ep]
        chance = chance_text(r["chance"])
        for rule, label in (("reference", "BirdNET"), ("best average rank", "Perch 2.0")):
            m = entry["models"][entry["models_by_rule"][rule]]["allocation_sensitivity"]
            achieved = m["achieved_stranger_acceptance_calibrated_at_0.10"]
            rows.append(open_row(NAMES[ep], label, m, chance, achieved))
        if ep == "great-tit":
            rows.append(
                open_row(
                    NAMES[ep],
                    "BirdNET, song removed",
                    v5["masked-birdnet-v2.4"],
                    chance,
                    r["v5_achieved"]["masked-birdnet-v2.4"],
                )
            )
            rows.append(
                open_row(
                    NAMES[ep],
                    "Year and nest box only, no audio",
                    v5["acquisition-context"],
                    chance,
                    r["v5_achieved"]["acquisition-context"],
                )
            )
    out.append(
        "**Table 6.** Re-ID with unfamiliar animals (strangers) present: medians over 16 "
        "allocations of individuals to roles, with ranges.\n\n"
        + table(
            [
                "Dataset",
                "Embedding",
                "Names correctly",
                "Chance for naming",
                "Separation",
                "Equal error rate",
                "Accepted and named, test-derived threshold",
                "Accepted and named, calibrated threshold",
                "Strangers accepted, calibrated threshold",
            ],
            rows,
        )
        + "\n\n*Notes:* Names correctly: proportion of known individuals' test calls assigned "
        "to the correct individual among the known individuals of the allocation, with no "
        "strangers present. Chance for naming: one over the number of known individuals in an "
        "allocation (a range where allocations differ). Separation: probability that a known "
        "individual's call outscores a stranger's (0.5, none). Accepted and named: proportion "
        "of known individuals' test calls accepted and assigned to the correct individual. "
        "Test-derived threshold: the best value over all thresholds that accept at most one "
        "test stranger in ten, set with the test strangers themselves, so it is an upper limit "
        "only when no more than 10% of strangers are accepted. Calibrated threshold: set on "
        "other strangers to accept one in ten, as would be done in practice; the last column "
        "gives the proportion of test strangers it actually accepted. Perch 2.0 had the best "
        "average rank in the closed-set comparison. The great tit rows without audio and with "
        "the song removed come from the same allocations. All 38 embeddings and controls are "
        "in Table S15."
    )
    return "\n\n".join(out) + "\n"


BACKGROUND_ORDER = [
    "chiffchaff-withinyear",
    "littleowl-acrossyear",
    "pipit-withinyear",
    "pipit-acrossyear",
    "chiffchaff-acrossyear",
]
assert set(BACKGROUND_ORDER) == set(BACKGROUND_ENDPOINTS)


def open_row(
    dataset: str, label: str, m: dict[str, Any], chance: str, achieved: dict[str, float]
) -> list[str]:
    return [
        dataset,
        label,
        span(m["closed_set_accuracy"]),
        chance,
        span(m["known_vs_unknown_auroc"]),
        f3(m["equal_error_rate"]["median"]),
        span(m["best_acceptance_at_0.10"]),
        span(m["acceptance_calibrated_at_0.10"]),
        span(achieved),
    ]


def chance_text(chance: dict[str, float]) -> str:
    low, high = chance["minimum"], chance["maximum"]
    return f3(low) if math.isclose(low, high) else f"{f3(low)} to {f3(high)}"


#: Datasets whose recordings almost never hold two individuals, so Diagnostic 3 cannot be read.
NO_SHARED_RECORDINGS = {"bat-acrosstreatment"}
#: Datasets whose paired-clip "recording" is not a recording (the penguin's test clips carry
#: only a calendar date, across nests).
NO_PAIRED_CLIP = {"bat-acrosstreatment", "penguin-acrossnight"}


def supporting_tables(b: Build) -> dict[str, str]:
    headline = b.ledger("headline/headline-numbers-v6.json")
    carried = b.ledger("carried/numbers.json")
    controls = b.ledger("recording-controls/recording-controls-numbers.json")
    opened = b.ledger("open-set/open-set-numbers.json")
    additional = b.ledger("additional/additional-numbers-v6.json")
    spectral = b.ledger("spectral/spectral-subtraction-numbers.json")
    s: dict[str, str] = {}

    rows = []
    for ep in ORDER:
        if ep not in carried["confound_index"]:
            continue
        axes = carried["confound_index"][ep]["axes"]
        for axis in sorted(axes):
            if axis in ("split", "condition"):
                continue
            a = axes[axis]["all"]
            rows.append(
                [
                    NAMES[ep],
                    AXIS_WORDS.get(axis, axis),
                    a["levels"],
                    f3(a["nmi"]),
                    f3(a["cramers_v"]),
                    f"{a['levels_with_two_or_more_identities']} of {a['levels']}",
                    f"{a['identities_on_two_or_more_levels']} of {a['identities']}",
                ]
            )
    s["S1"] = (
        "**Table S1.** How closely identity is tied to each recording variable stored "
        "with the clips, over all clips of each dataset.\n\n"
        + table(
            [
                "Dataset",
                "Recording variable",
                "Levels",
                "NMI",
                "Cramér's V",
                "Levels holding two or more individuals",
                "Individuals on two or more levels",
            ],
            rows,
        )
        + "\n\n*Notes:* NMI, mutual information divided by the smaller of the two entropies: "
        "0, the variable is unrelated to identity; 1, either determines the other. It is 1 "
        "both when every level holds one individual (each recording one animal) and when "
        "every individual sits on one level (each bird in one year cohort); the last two "
        "columns show which. A variable with one level carries no information. The Stowell "
        "et al. datasets store no recording variable within a split."
    )

    place = carried["place_control"]
    full = additional["great_tit_place"]
    tables_json = json.loads((b.build / "carried/tables.json").read_text())
    s["S2"] = (
        "**Table S2.** Great tit place control: for each bird of the 16-bird dataset, "
        "whether any nest box or recording contributed to both its enrolment and test "
        "clips, whether the years differ, and the distance between its enrolment and "
        "test nest boxes.\n\n"
        + relabel_place(tables_json["place"])
        + f"\n\n*Notes:* Bird codes are arbitrary; ring numbers are not published. "
        f"Distances ranged from {place['metres_between_nestboxes']['minimum']:.1f} to "
        f"{place['metres_between_nestboxes']['maximum']:.1f} m (median "
        f"{place['metres_between_nestboxes']['median']:.1f} m). In the 50-bird dataset, "
        f"no recording contributed to both sides, but {full['shared_nestbox']} of "
        f"{full['individuals']} birds were enrolled and tested at the same nest box; "
        f"distances ranged from {full['distance_metres']['minimum']:.1f} to "
        f"{full['distance_metres']['maximum']:.1f} m (median "
        f"{full['distance_metres']['median']:.1f} m)."
    )

    rows = []
    for ep in ORDER:
        h = headline[f"headline.{HEADLINE_STEM[ep]}"]
        rows.append(s3_row(NAMES[ep], h, ep))
    g = b.result("v6-headline-great-tit-full.json")
    e = g["evaluation"]
    rows.append(
        [
            NAMES["great-tit-full"],
            g["identities"] if isinstance(g["identities"], int) else len(g["identities"]),
            f"{e['enrollment_calls']:,} / {e['query_calls']:,}",
            iv(e["classification"]["accuracy"], e["identity_block_bootstrap_accuracy_95"]),
            f"{f3(e['hierarchical_bootstrap']['accuracy_95'][0])} to "
            f"{f3(e['hierarchical_bootstrap']['accuracy_95'][1])}",
            f3(g["uniform_chance"]),
            f3(g["majority_class_rate"]),
            f3(e["permutation_upper_tail"]["percentile_95"]),
            p4(e["permutation_control"]["p_value_plus_one"]),
            "n/a",
            "n/a",
            "n/a",
            "n/a",
        ]
    )
    s["S3"] = (
        "**Table S3.** Closed-set re-ID with BirdNET and the kernel ridge classifier on "
        "every dataset, all quantities.\n\n"
        + table(
            [
                "Dataset",
                "n",
                "Enrolment / test clips",
                "Re-ID accuracy (95% interval)",
                "Three-stage interval",
                "Uniform chance",
                "Majority class",
                "Permutation 95th percentile",
                "p",
                "Within-recording 95th percentile (p)",
                "Paired-clip difference",
                "Standardised difference",
                "Equal error rate",
            ],
            rows,
        )
        + "\n\n*Notes:* As Table 2. Paired-clip test: mean cosine similarity of animal pairs "
        "(one individual, different recordings) minus that of recording pairs (different "
        "individuals, one recording), its standardised value and the equal error rate for "
        "telling the two kinds of pair apart from a single pair. n/a: the dataset has no "
        "recording holding two individuals, or does not identify recordings. The fruit bat "
        "has only five pairs of test clips from two bats in one recording and 16 enrolment "
        "clips that the within-recording test can shuffle, so neither test is reported. The "
        "little penguin's test clips record only a calendar date, across nests, so a pair "
        "from one date is not a pair from one recording."
    )

    rook_rows = []
    words = {
        "across-year": "Enrolled in 2020, tested in 2021",
        "across-year-capped-90": "The same, enrolment capped at 90 clips per bird",
        "across-year-random-fold": "The same clips, divided at random within each bird",
        "across-day-2020": "Enrolled and tested on different 2020 days",
        "across-day-2020-ch0-to-ch1": "The same calls, enrolled on channel 0, tested on channel 1",
    }
    for stem, word in words.items():
        r = carried[f"rook.{stem}"]
        enrol, test = (int(v) for v in r["clips"].split(" / "))
        rook_rows.append(
            [
                word,
                r["individuals"],
                f"{enrol:,} / {test:,}",
                iv(r["accuracy"], r["interval"]),
                f"{f3(r['three_stage'][0])} to {f3(r['three_stage'][1])}",
                f3(r["permutation_95"]),
                p4(r["permutation_p"]),
                p4(r["within_p"]),
                f3(r["trials_standardised"]),
                f3(r["trials_eer"]),
            ]
        )
    s["S4"] = (
        "**Table S4.** The rook under five divisions, with BirdNET.\n\n"
        + table(
            [
                "Division",
                "n",
                "Enrolment / test clips",
                "Re-ID accuracy (95% interval)",
                "Three-stage interval",
                "Permutation 95th percentile",
                "p",
                "Within-recording p",
                "Standardised paired-clip difference",
                "Equal error rate",
            ],
            rook_rows,
        )
        + "\n\n*Notes:* Channel 0 throughout unless stated. The channel index is not the same "
        "physical microphone on every recording day."
    )

    rows = []
    for ep in BACKGROUND_ORDER:
        for model in ["birdnet-v2.4"] + MODELS:
            block = controls[f"{ep}.controls.{full_name(controls, ep, model)}"]["four_pairings"]
            rows.append(
                [NAMES[ep], DISPLAY.get(model, "BirdNET v2.4")]
                + [pairing_cell(block[k], "") for k in PAIRINGS]
            )
    s["S5"] = (
        "**Table S5.** Re-ID from calls and from background recordings for all 38 "
        "embeddings and controls, each at its layer chosen by the Fisher rule.\n\n"
        + table(["Dataset", "Embedding"] + [SHORT_PAIR[k] for k in PAIRINGS], rows)
        + "\n\n*Notes:* C→B: classifier fitted on calls and tested on backgrounds; the "
        "other columns likewise. Each cell: re-ID accuracy (95% interval over "
        "individuals, 10,000 draws); permutation p (9,999 shuffles), unadjusted. Entries "
        "whose B→B p is below 0.05 after Holm's correction over the 38 entries of a dataset: "
        + "; ".join(
            f"{NAMES[ep]} {b.review['counts']['background'][ep]['pairings']['background_gallery_background_query']['holm_below_0.05']}"
            for ep in BACKGROUND_ORDER
        )
        + " (unadjusted: "
        + "; ".join(
            f"{b.review['counts']['background'][ep]['pairings']['background_gallery_background_query']['unadjusted_below_0.05']}"
            for ep in BACKGROUND_ORDER
        )
        + "; about 2 of 38 expected by chance)."
    )

    rows = []
    for ep in BACKGROUND_ORDER:
        for model in ["birdnet-v2.4"] + MODELS:
            c = controls[f"{ep}.controls.{full_name(controls, ep, model)}"]["challenge"][
                "own_vs_other"
            ]
            rows.append(
                [NAMES[ep], DISPLAY.get(model, "BirdNET v2.4")]
                + [
                    iv(c[k]["own_minus_other_macro_recall"], c[k]["identity_block_difference_95"])
                    for k, _ in LEVELS
                ]
            )
    s["S6"] = (
        "**Table S6.** Effect of adding a background recording to test clips, for all "
        "38 embeddings and controls: own-donor minus other-donor balanced accuracy.\n\n"
        + table(["Dataset", "Embedding"] + [f"Background {w}" for _, w in LEVELS], rows)
        + "\n\n*Notes:* Background level relative to the test clip. Positive values mean "
        "the classifier's answers follow the background; negative values mean a clip mixed "
        "with its own individual's test-year background was named less accurately than one "
        "mixed with another individual's. Interval over individuals, 10,000 draws."
    )

    s["S7"] = donor_table(b.review["donors"])

    sens = additional["sensitivity"]
    raw = b.result("v6-sensitivity-floor.json")
    rows = [
        [
            word,
            iv(sens[k]["accuracy"], sens[k]["interval_95"]),
            p4(sens[k]["permutation_p"]),
            f3(sens[k]["chance"]),
            f3(raw["levels"][k]["majority_class_rate"]),
        ]
        for k, word in (
            ("+10db", "Call 10 dB above background"),
            ("+0db", "Equal level"),
            ("-10db", "Call 10 dB below background"),
        )
    ]
    s["S8"] = (
        "**Table S8.** Planted-signal test: calls of ten within-year chiffchaffs mixed "
        "into the across-year chiffchaff backgrounds, one bird per territory, scored "
        "with BirdNET.\n\n"
        + table(
            ["Mixture", "Re-ID accuracy (95% interval)", "p", "Uniform chance", "Majority class"],
            rows,
        )
        + "\n\n*Notes:* Interval over individuals, 10,000 draws; p from 9,999 label "
        "shuffles."
    )

    rows = []
    arms = {"untouched": "Original", "stratified-background-augmentation": "Augmented"}
    for ep in ("chiffchaff-withinyear", "littleowl-acrossyear"):
        r = carried[f"remedy.{ep}"]
        for arm, word in arms.items():
            rows.append(
                [NAMES[ep], "Background augmentation", word]
                + [
                    iv(r[arm][k]["accuracy"], r[arm][k]["interval"]) + f"; {p4(r[arm][k]['p'])}"
                    for k in (
                        "foreground_to_foreground",
                        "background_to_background",
                        "foreground_to_background",
                        "background_to_foreground",
                    )
                ]
            )
    aug = additional["augmentation"]
    for arm, word in arms.items():
        c = aug[arm]["pairings_by_head"]["kernel ridge to one-hot"]
        rows.append(
            [NAMES["pipit-withinyear"], "Background augmentation", word]
            + [
                iv(c[k]["accuracy"], c[k]["identity_block_bootstrap_accuracy_95"])
                + f"; {p4(c[k]['permutation_p'])}"
                for k in (
                    "foreground_to_foreground",
                    "background_to_background",
                    "foreground_to_background",
                    "background_to_foreground",
                )
            ]
        )
    for ep in BACKGROUND_ORDER:
        sp = spectral[ep]
        for arm, word in (
            ("original", "Original"),
            ("alpha_1", "Subtracted once"),
            ("alpha_2", "Subtracted twice"),
        ):
            rows.append(
                [NAMES[ep], "Spectral subtraction", word]
                + [
                    iv(sp[arm][k]["accuracy"], sp[arm][k]["interval_95"])
                    + f"; {p4(sp[arm][k]['label_permutation_p'])}"
                    for k in PAIRINGS
                ]
            )
    s["S9"] = (
        "**Table S9.** Background augmentation and spectral subtraction with BirdNET, "
        "scored through the four pairings.\n\n"
        + table(["Dataset", "Remedy", "Arm", "C→C", "B→B", "C→B", "B→C"], rows)
        + "\n\n*Notes:* Each cell: re-ID accuracy (95% interval over individuals, 10,000 "
        "draws); permutation p (9,999 shuffles). Background augmentation changes only the "
        "enrolment calls, so the two pairings fitted on backgrounds are unchanged. Spectral "
        "subtraction removes each individual's mean background spectrum, once or twice, from "
        "all its clips. The original arm was scored with its own draws of individuals, so its "
        "interval bounds can differ slightly from those in Table 4."
    )

    rows = []
    for ep in BACKGROUND_ORDER:
        norms = controls[f"{ep}.as_norm"]
        for model in ["birdnet-v2.4"] + MODELS:
            n = norms[full_name(norms, None, model)]
            rows.append(
                [NAMES[ep], DISPLAY.get(model, "BirdNET v2.4")]
                + [
                    f"{f3(n[c]['raw_class_mean']['accuracy'])} to {f3(n[c]['as_norm']['accuracy'])}"
                    for c in ("calls_scored", "backgrounds_scored")
                ]
            )
    s["S10"] = (
        "**Table S10.** Adaptive score normalisation with the class-mean classifier: "
        "re-ID accuracy before and after, for all 38 embeddings and controls.\n\n"
        + table(["Dataset", "Embedding", "Calls tested", "Backgrounds tested"], rows)
        + "\n\n*Notes:* Backgrounds tested: classifier fitted on backgrounds and tested "
        "on backgrounds. Intervals are in the numerical ledger."
    )

    comp = carried["compensation"]
    rows = [
        [
            TREATMENT[c["treatment"]],
            setting(c["parameter"]),
            f3(c["identity"]),
            f3(c["day"]),
            f3(c["day_floor"]),
        ]
        for c in comp["bat_settings"]
    ]
    s["S11"] = (
        "**Table S11.** Session compensation on the fruit bat with BirdNET: re-ID "
        "accuracy and how well the recording day could still be predicted, for every "
        "setting specified in advance.\n\n"
        + table(
            ["Method", "Setting", "Re-ID accuracy", "Day predicted", "Day, majority rate"], rows
        )
        + "\n\n"
        + table(
            ["Method", "Setting", "Classifier", "Re-ID accuracy (95% interval)"],
            [
                [
                    TREATMENT[c["treatment"]],
                    setting(c["parameter"]),
                    HEAD[c["head"]],
                    iv(c["accuracy"], c["interval"]),
                ]
                for c in comp["owl"]
            ],
        )
        + "\n\n*Notes:* Upper: fruit bat. Day predicted: accuracy of a classifier fitted on all "
        "but one bat and tested on the one left out, for each bat in turn. Lower: "
        "within-class covariance normalisation on the little owl with three classifiers. "
        "Removing 0 directions and shrinkage 1.0 leave the embedding unchanged and repeat "
        "the row without compensation. Spearman correlation between re-ID accuracy and day "
        f"prediction over the {b.review['bat']['distinct']} distinct bat settings: "
        f"{f3(b.review['bat']['spearman_distinct'][0])} "
        f"(p = {p4(b.review['bat']['spearman_distinct'][1])}); without the two "
        f"session-mean rows, {f3(b.review['bat']['spearman_without_mean_subtraction'][0])} "
        f"(p = {p4(b.review['bat']['spearman_without_mean_subtraction'][1])}). Subtracting "
        "each day's mean over all bats places a bat's clips opposite the other bats' clips of "
        "the same day, so a day classifier fitted on the other bats is systematically wrong; "
        "the day prediction of 0.016 is a result of that design."
    )
    s.update(more_supporting(b, carried, opened, tables_json))
    return s


# The five-bird rook of the earlier run, replaced in this paper by the 11-bird rook.
SUPERSEDED = {"rookid"}

OLD_NAMES = {
    "rookid": "Rook, five birds, one call type",
    "zebra": NAMES["zebra-finch"],
    "zebra-finch": NAMES["zebra-finch"],
    "great-tit": NAMES["great-tit"],
    "littleowl": NAMES["littleowl-acrossyear"],
    "littleowl-acrossyear": NAMES["littleowl-acrossyear"],
    "chiffchaff-withinyear": NAMES["chiffchaff-withinyear"],
    "chiffchaff-acrossyear": NAMES["chiffchaff-acrossyear"],
    "pipit-withinyear": NAMES["pipit-withinyear"],
    "pipit-acrossyear": NAMES["pipit-acrossyear"],
    "birdpark-juv01": NAMES["birdpark-juv01"],
    "birdpark-juv03": NAMES["birdpark-juv03"],
    "bat-acrosstreatment": NAMES["bat-acrosstreatment"],
}
BIRDNET_LAYER = {
    "stage1": "stage 1",
    "stage2": "stage 2",
    "stage3": "stage 3",
    "stage4": "stage 4",
    "post_activation": "post-activation",
    "post_convolution": "post-convolution",
    "embedding": "embedding",
}
POOL = {"mean": "mean", "mean_std": "mean and spread", "sequence": "sequence"}
REFERENCE = {
    "perch-v2": "van Merriënboer et al. (2025)",
    "perch-bird": "Ghani et al. (2023)",
    "surfperch": "Williams et al. (2025)",
    "birdnet-v3-preview": "developer preview, as supplied by Bacpipe 1.3.5 (Kather et al. 2026)",
    "esp-aves2-sl-beats-all": "Miron et al. (2026)",
    "esp-aves2-effnetb0-all": "Miron et al. (2026)",
    "avesecho-passt": "Ghani et al. (2025)",
    "audioprotopnet": "Heinrich et al. (2025)",
    "convnext-birdset": "Rauch et al. (2025b)",
    "birdmae": "Rauch et al. (2025a)",
    "protoclr": "Moummad et al. (2026)",
    "rcl-fs-bsed": "Moummad et al. (2024)",
    "birdaves": "Hagiwara (2023)",
    "aves": "Hagiwara (2023)",
    "naturebeats": "Robinson et al. (2025)",
    "biolingual": "Robinson et al. (2024)",
    "beats": "Chen et al. (2023)",
    "audiomae": "Huang et al. (2022)",
    "vggish": "Hershey et al. (2017)",
}


SPEECH_REFERENCE = {
    "spkrec-ecapa-voxceleb": "Desplanques et al. (2020); Ravanelli et al. (2021)",
    "spkrec-resnet-voxceleb": "Villalba et al. (2020); Ravanelli et al. (2021)",
    "spkrec-xvect-voxceleb": "Snyder et al. (2018); Ravanelli et al. (2021)",
    "wav2vec2-base": "Baevski et al. (2020)",
    "wav2vec2-large-robust": "Hsu et al. (2021b)",
    "wav2vec2-xls-r-300m": "Babu et al. (2022)",
    "wav2vec2-conformer-rope-large": "Gulati et al. (2020); Wang et al. (2020)",
    "mms-300m": "Pratap et al. (2024)",
    "hubert-base-ls960": "Hsu et al. (2021a)",
    "hubert-large-ll60k": "Hsu et al. (2021a)",
    "data2vec-audio-base-100h": "Baevski et al. (2022)",
    "data2vec-audio-base-960h": "Baevski et al. (2022)",
    "wavlm-base-plus": "Chen et al. (2022a)",
    "wavlm-large": "Chen et al. (2022a)",
    "unispeech-sat-base-plus": "Chen et al. (2022b)",
    "xeus": "Chen et al. (2024)",
}


def birdnet_layer(name: str) -> str:
    layer, _, pool = name.partition(".")
    return f"{BIRDNET_LAYER.get(layer, layer)}, {POOL.get(pool, pool)}"


def candidate(name: str | None) -> str:
    import re

    if name is None:
        return "n/a"
    if name.startswith("clip-"):
        return "summary"
    m = re.search(r"-x(\d)-l(\d+)$", name)
    if m:
        speed = "full speed" if m.group(1) == "1" else f"1/{m.group(1)} speed"
        return f"layer {int(m.group(2))}, {speed}"
    m = re.search(r"-x(\d)$", name)
    if m:
        return "embedding" if m.group(1) == "1" else f"embedding, 1/{m.group(1)} speed"
    m = re.match(r"block (\d+)$", name)
    if m:
        return f"block {int(m.group(1))}"
    return name.replace("_", " ")


def relabel_rows(text: str, first_column: dict[str, str], cells: dict[str, str]) -> str:
    lines = text.split("\n")
    out = lines[:2]
    for line in lines[2:]:
        parts = [p.strip() for p in line.strip("|").split("|")]
        parts[0] = first_column.get(parts[0], parts[0])
        parts = [cells.get(p, p) for p in parts]
        out.append("| " + " | ".join(parts) + " |")
    return "\n".join(out)


def more_supporting(
    b: Build, carried: dict[str, Any], opened: dict[str, Any], tables_json: dict[str, str]
) -> dict[str, str]:
    s: dict[str, str] = {}
    provenance = json.loads((b.build / "provenance/model-provenance-v6.json").read_text())["models"]
    coverage = json.loads((b.build / "coverage/candidate-coverage.json").read_text())["endpoints"]
    rows = []
    for model in MODELS:
        if model.startswith("clip-"):
            continue
        entries = [ep[m] for ep in coverage.values() for m in ep if short(m) == model]
        counts = sorted({e["candidate_count"] for e in entries})
        count = str(counts[0]) if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"
        prov = [v for k, v in provenance.items() if short(k) == model]
        if prov:
            p = prov[0]
            rows.append(
                [
                    DISPLAY[model],
                    "animal or general sound",
                    REFERENCE[model],
                    f"{p['sample_rate']:,}",
                    f"{p['sample_rate'] / 2000:g}",
                    f"{p['window_seconds']:g}",
                    count,
                    "1",
                ]
            )
        else:
            full = [m for ep in coverage.values() for m in ep if short(m) == model][0]
            speeds = sorted(
                {
                    c["slowdown"]
                    for e in entries
                    for c in e["candidates"].values()
                    if c["slowdown"] is not None
                }
            )
            rows.append(
                [
                    DISPLAY[model],
                    "human speech",
                    f"{SPEECH_REFERENCE[model]}; `{full}`",
                    "16,000",
                    "8; 16 and 24 at slowed playback",
                    "whole clip",
                    count,
                    ", ".join(str(x) for x in speeds),
                ]
            )
    s["S12"] = (
        "**Table S12.** The 35 neural networks compared with BirdNET v2.4.\n\n"
        + table(
            [
                "Neural network",
                "Group",
                "Source",
                "Input sample rate (Hz)",
                "Highest frequency analysed (kHz)",
                "Window (s)",
                "Candidate embeddings per dataset",
                "Playback slowing",
            ],
            rows,
        )
        + "\n\n*Notes:* Speech networks are identified by their method papers and published "
        "checkpoint names. "
        "Window: the input length each clip was cut into; speech networks read whole clips, "
        "or 60-s windows where a clip exceeded the graphics card's memory. Candidate "
        "embeddings: layers or blocks read, times playback speeds, where computed on that "
        "dataset. Playback slowing: 1, the recording's own speed; 2 and 3, half and one third "
        "of it. Highest frequency analysed: half the input sample rate, the most a network can "
        "represent; for speech networks, 8 kHz at the recording's own speed and 16 and 24 kHz "
        "at half and one third of it. BirdNET v2.4 reads 48 kHz audio. Sample rates and "
        "windows come from a record made by reloading each network. The speaker-verification "
        "checkpoint of WavLM Base+ (microsoft/wavlm-base-plus-sv) was also measured; the layers "
        "read have the weights of WavLM Base+ and every value was identical, so it is not "
        "counted separately. BirdNET v3 (preview) is the ONNX file Bacpipe 1.3.5 supplies "
        # SHA-256 of models/bacpipe/birdnet_v3/model.onnx as recorded in v6-weights.json.
        "(SHA-256 6f58d7ffa4c33bf49c8c67ac27bc5265a940a139cb67254477997bad41efc16d)."
    )

    review = b.review["comparison"]
    per = review["per_endpoint"]
    rows_by_ep = []
    for ep in ORDER_COMPARISON:
        rows = []
        for model in MODELS:
            cells = [DISPLAY[model]]
            for rule in ("Fisher", "held_out"):
                cells += rule_cells(per[ep].get(rule), model)
            rows.append(cells)
        ref = per[ep]["Fisher"]["paired"]
        rows_by_ep.append(
            f"*{NAMES[ep]}*: BirdNET v2.4 "
            f"{iv(ref['reference_accuracy'], ref['reference_accuracy_95'])}.\n\n"
            + table(
                [
                    "Neural network",
                    "Fisher rule: layer, re-ID accuracy",
                    "Difference from BirdNET",
                    "Holm p",
                    "Held-out rule: layer, re-ID accuracy",
                    "Difference from BirdNET",
                    "Holm p",
                ],
                rows,
            )
        )
    s["S13"] = (
        "**Table S13.** Each neural network and control against BirdNET v2.4 on every "
        "dataset, under the two layer rules specified in advance.\n\n"
        + "\n\n".join(rows_by_ep)
        + "\n\n*Notes:* Re-ID accuracy with a 95% interval over individuals; difference "
        "from BirdNET with a paired interval over the same 10,000 draws of individuals; "
        "two-sided p from the share of the 10,000 draws on each side of zero, adjusted by "
        "Holm's method over the 37 comparisons on that dataset and rule. When no draw crossed "
        "zero the adjusted p is below 37 × 2 / 10,000, shown as < 0.0074. BirdNET's interval "
        "comes from the same paired draws, so its bounds can differ slightly from those in "
        "Table 2. n/a: the held-out rule does not apply, because the dataset has fewer than two "
        "enrolment sessions per individual, and on such datasets it would choose what the "
        "Fisher rule chose."
    )

    summary = review["held_out_changes"]
    counts = review["counts"]
    ranks = []
    for label, key in (
        ("Fisher, 13 datasets", "Fisher"),
        ("Held-out, 6 datasets", "held_out"),
        ("Own final output, 13 datasets", "final"),
    ):
        r = review["ranks"][key]
        ranks.append(
            [
                label,
                len(r["representations"]),
                f"{r['friedman_chi_square']:.1f}",
                "< 0.0001" if r["friedman_p"] < 0.0001 else f"{r['friedman_p']:.4f}",
                f"{r['kendall_w']:.2f}",
                f"{r['nemenyi_critical_difference_0.05']:.1f}",
            ]
        )
    average = {key: review["ranks"][key]["average_rank"] for key in ("Fisher", "held_out", "final")}
    order = sorted(average["Fisher"], key=lambda m: average["Fisher"][m])
    listing = [
        [DISPLAY.get(m, "BirdNET v2.4")] + [f"{average[k][m]:.2f}" for k in average] for m in order
    ]
    f, h, o = counts["Fisher"], counts["held_out"], counts["final"]
    s["S14"] = (
        "**Table S14.** Consistency of the ranking of embeddings across datasets under the "
        "three layer rules, and each entry's average rank.\n\n"
        + table(
            [
                "Layer rule",
                "Entries ranked",
                "Friedman χ²",
                "p",
                "Kendall's W",
                "Nemenyi critical difference",
            ],
            ranks,
        )
        + "\n\n"
        + table(
            [
                "Embedding",
                "Average rank, Fisher rule",
                "Average rank, held-out rule",
                "Average rank, own final output",
            ],
            listing,
        )
        + f"\n\n*Notes:* Rank 1 is the highest re-ID accuracy on a dataset. Held-out rule: the "
        f"six datasets where it applies. Own final output: specified after the review, so "
        f"exploratory. Comparisons with BirdNET after Holm's correction: Fisher rule "
        f"{f['above']} of {f['comparisons']} above and {f['below']} below; held-out rule "
        f"{h['above']} of {h['comparisons']} above and {h['below']} below; own final output "
        f"{o['above']} of {o['comparisons']} above and {o['below']} below. On the six datasets, "
        f"the held-out rule changed the chosen layer in {summary['changed']} of "
        f"{summary['comparisons']} comparisons, raising re-ID accuracy in {summary['higher']} "
        f"and lowering it in {summary['lower']}."
    )

    rows = []
    for ep in (
        "great-tit",
        "great-tit-full",
        "littleowl-acrossyear",
        "rookid-full-width",
        "cockatoo-fold1",
    ):
        models = opened[ep]["models"]
        for model in ["birdnet-v2.4"] + MODELS:
            key = [k for k in models if short(k) == model][0]
            m = models[key]["allocation_sensitivity"]
            rows.append(
                [
                    NAMES[ep],
                    DISPLAY.get(model, "BirdNET v2.4"),
                    span(m["closed_set_accuracy"]),
                    span(m["known_vs_unknown_auroc"]),
                    span(m["acceptance_calibrated_at_0.10"]),
                    span(m["achieved_stranger_acceptance_calibrated_at_0.10"]),
                    span(m["balanced_accuracy_known_calibrated_at_0.10"]),
                    span(m["balanced_accuracy_unknown_calibrated_at_0.10"]),
                    span(m["geometric_mean_known_unknown_calibrated_at_0.10"]),
                ]
            )
    s["S15"] = (
        "**Table S15.** Re-ID with strangers present for all 38 embeddings and "
        "controls: medians over 16 allocations, with ranges.\n\n"
        + table(
            [
                "Dataset",
                "Embedding",
                "Names correctly",
                "Separation",
                "Accepted and named, calibrated",
                "Strangers accepted, calibrated",
                "Balanced accuracy, known",
                "Balanced accuracy, strangers",
                "Geometric mean",
            ],
            rows,
        )
        + "\n\n*Notes:* Calibrated: threshold set on other strangers to admit one in ten; "
        "strangers accepted: the proportion of test strangers it actually admitted. "
        "Balanced accuracy on known individuals: mean over individuals of the proportion of "
        "their calls accepted and correctly named; on strangers: proportion rejected. "
        "Geometric mean of the two, as in the AnimalCLEF 2025 benchmark (Adam et al. 2025)."
    )

    frozen = [
        [
            OLD_NAMES.get(r[0], r[0]),
            birdnet_layer(r[1]),
            f3(r[2]),
            r[3],
            f3(r[4]),
            r[5].replace(r[5].split(" (")[0], birdnet_layer(r[5].split(" (")[0])),
        ]
        for r in carried["frozen_layers"]
        if r[0] not in SUPERSEDED
    ]
    learned = [
        [OLD_NAMES.get(r[0], r[0]), r[1], f3(r[2]), f3(r[3]), f3(r[4])]
        for r in carried["learned_combination"]
        if r[0] not in SUPERSEDED
    ]
    sequence = [
        [OLD_NAMES.get(r[0], r[0]), birdnet_layer(r[1]), r[2], f3(r[3]), f3(r[4]), f3(r[5])]
        for r in carried["sequence_metric"]
        if r[0] not in SUPERSEDED
    ]
    s["S16"] = (
        "**Table S16.** BirdNET's internal layers, a learned combination of them, and "
        "learned pooling over time.\n\n"
        + table(
            [
                "Dataset",
                "Layer chosen by the Fisher rule",
                "Its re-ID accuracy",
                "Its adjusted p",
                "Final embedding",
                "Best layer chosen with the test labels",
            ],
            frozen,
        )
        + "\n\n"
        + table(["Dataset", "Embedding", "Kernel ridge", "Class mean", "Nearest clip"], learned)
        + "\n\n"
        + table(
            [
                "Dataset",
                "Layer",
                "Individuals scored",
                "Layer averaged over time",
                "Untrained sequence average",
                "Learned sequence pooling",
            ],
            sequence,
        )
        + "\n\n*Notes:* Upper: the 14 candidates of Appendix S1.13, with the permutation p "
        "(9,999 shuffles) multiplied by the number of distinct candidates (12, since two of "
        "the 14 repeat others) and capped at one; the best layer chosen with the "
        "test labels cannot be chosen in practice. Middle: learned combination against the "
        "final embedding alone, the same projection fitted to the final embedding, and the "
        "seven layers joined; no interval was computed on the differences. Lower: learned "
        "pooling scored on individuals it never saw (3 to 7 per dataset), so its values are "
        "not comparable in scale with Table 2."
    )

    whale = b.review["whale"]
    s["S17"] = (
        "**Table S17.** The four pairings on North Atlantic right whale upcalls and tag "
        "noise from 11 whales (data and frequency shift of Tolkova et al. 2026), with "
        "BirdNET.\n\n"
        + relabel_rows(
            tables_json["whale"]
            .replace("Enrolment Fisher ratio", "Fisher ratio on enrolment clips")
            .replace("calls on calls (accuracy / p)", "C→C (accuracy / p)")
            .replace("backgrounds on backgrounds (accuracy / p)", "B→B (accuracy / p)")
            .replace("calls scored on backgrounds (accuracy / p)", "C→B (accuracy / p)")
            .replace("backgrounds scored on calls (accuracy / p)", "B→C (accuracy / p)"),
            {},
            {},
        )
        + "\n\n*Notes:* Each whale carried one tag deployment, so enrolment and test clips come "
        "from one recording. The shift moves the 50 to 500 Hz upcall band upward before "
        "BirdNET reads it, identically for calls and noise. The shift for the result was "
        "specified in advance as the one with the highest Fisher ratio on the enrolment "
        "calls (10,000 Hz). p from 9,999 label shuffles. After Holm's correction over the 11 "
        f"shifts, B→C exceeded chance at {whale['background_to_foreground']['holm_below_0.05']} "
        f"shifts and C→B at {whale['foreground_to_background']['holm_below_0.05']}, and neither "
        "at 10,000 Hz. Data: github.com/avokloti/narw-acoustic-identification, commit 3cad65d, "
        "which declares no licence; no clip is redistributed. No identity claim is made from "
        "this dataset."
    )

    rows = []
    information = {ep: b.ledger(f"information/{ep}-numbers.json")["models"] for ep in ENDPOINTS}
    for model in ["birdnet-v2.4"] + [m for m in MODELS if not m.startswith("clip-")]:
        cells = [DISPLAY.get(model, "BirdNET v2.4")]
        for ep in ENDPOINTS:
            entry = [v for k, v in information[ep].items() if short(k) == model][0]
            bits = entry["fisher"]["hs_bits"]
            cells.append("n/a" if bits is None else f"{bits:.2f}")
        rows.append(cells)
    s["S18"] = (
        "**Table S18.** Beecher's information statistic (bits) for each neural network "
        "at its layer chosen by the Fisher rule.\n\n"
        + table(["Neural network"] + [NAMES[ep] for ep in ENDPOINTS], rows)
        + "\n\n*Notes:* Computed on all call clips of each dataset from principal components "
        "(Appendix S1.7). It measures how well clips group by label, whether the grouping "
        "comes from the animal or the recording."
    )

    repro = json.loads((b.build / "reproduction/historical-reproduction.json").read_text())
    rows = [
        [NAMES[ep], r["candidates_expected"], r["candidates_compared"], r["identical"]]
        for ep, r in repro["endpoints"].items()
    ]
    s["S19"] = (
        "**Table S19.** BirdNET, the 16 speech network checkpoints of the earlier run and two "
        "controls, run twice: candidate embeddings compared on each dataset.\n\n"
        + table(["Dataset", "Candidates in the first run", "Compared", "Identical"], rows)
        + "\n\n*Notes:* Identical: the same re-ID accuracy and the same right or wrong answer "
        "on every test clip. The two candidates that differed (Appendix S1.3) were not chosen "
        "by either layer rule, and every reported embedding was identical."
    )

    s["S20"] = CORPORA

    within = b.review["counts"]["within_recording"]
    shown = ["birdpark-juv01", "birdpark-juv03", "rookid-full-width", "zebra-finch"]
    rows = []
    for model in ["birdnet-v2.4"] + MODELS:
        cells = [DISPLAY.get(model, "BirdNET v2.4")]
        for ep in shown:
            r = within[ep]["rows"][model]
            cells.append(
                f"{f3(r['accuracy'])}; {f3(r['within_recording_95'])}; {p4(r['within_recording_p'])}"
            )
        rows.append(cells)
    s["S21"] = (
        "**Table S21.** Within-recording permutation test for all 38 embeddings and controls "
        "on the datasets in which recordings hold several individuals.\n\n"
        + table(["Embedding"] + [NAMES[ep] for ep in shown], rows)
        + "\n\n*Notes:* Each cell: re-ID accuracy; 95th percentile of the within-recording "
        "test; its p (9,999 shuffles of labels among clips of the same recording, or, for the "
        "zebra finch recorded singly, the same day). Entries with p < 0.05: "
        + "; ".join(f"{NAMES[ep]} {within[ep]['below_0.05']}" for ep in shown)
        + ". Each embedding at its layer chosen by the Fisher rule."
    )

    rows_by_ep = []
    for ep in ORDER_COMPARISON:
        rows = [[DISPLAY[model]] + rule_cells(per[ep]["final"], model) for model in MODELS]
        ref = per[ep]["final"]["paired"]
        rows_by_ep.append(
            f"*{NAMES[ep]}*: BirdNET v2.4 "
            f"{iv(ref['reference_accuracy'], ref['reference_accuracy_95'])}.\n\n"
            + table(
                [
                    "Neural network",
                    "Output read, re-ID accuracy",
                    "Difference from BirdNET",
                    "Holm p",
                ],
                rows,
            )
        )
    g = review["final_vs_fisher"]
    animal = g["animal networks with several layers"]
    speech = g["speech networks with several layers"]
    s["S22"] = (
        "**Table S22.** Exploratory analysis, specified after the fourth round of review: each "
        "neural network read at its own final output against BirdNET v2.4.\n\n"
        + "\n\n".join(rows_by_ep)
        + "\n\n*Notes:* Own final output: the output embedding for networks trained on animal "
        "or general sound, the last transformer layer at the recording's own speed for speech "
        "networks, and the output at the recording's own speed for speaker networks; networks "
        "with one embedding are as in Table S13. Difference and Holm p as in Table S13. Against "
        f"the layer chosen by the Fisher rule, the final output was higher in {animal['final_higher']} "
        f"of {animal['pairs']} network-dataset pairs for the ten animal networks with several "
        f"layers (median change {signed(animal['median'])}) and in {speech['final_higher']} of "
        f"{speech['pairs']} for the 13 speech networks with several layers (median "
        f"{signed(speech['median'])})."
    )
    return s


ORDER_COMPARISON = [e for e in ENDPOINTS if e != "great-tit-full"]


def signed(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")


def rule_cells(block: dict[str, Any] | None, model: str) -> list[str]:
    if block is None or model not in block["paired"]["comparisons"]:
        return ["n/a", "n/a", "n/a"]
    comp = block["paired"]["comparisons"][model]
    layer = block["selection"][model]["selected"]
    return [
        f"{candidate(layer)}: {iv(comp['accuracy'], comp['accuracy_95'])}",
        iv(comp["difference_from_reference"], comp["difference_95"]),
        p_draws(comp["p_holm"], block["paired"]["zero_p_bound"]),
    ]


CORPORA = """**Table S20.** Datasets screened and not used, with the reason.

| Dataset | Reported scale | Reason not used |
|---|---|---|
| Roroa (great spotted kiwi, *Apteryx maxima*), Bedoya & Molles (2021) | 849 calls, 30 individuals | Call labels derive from the nest associated with each recorder and the published workflow divides calls at random, so nest, recorder, location and identity are not independently crossed. |
| Wild zebra finches, Chauhan et al. (2025) | 2,915 clips, 173 individuals | Identity labels were assigned from distinctive song exemplars, with a median of eight clips per individual; too few to hold a group of strangers back. |
| House wren, Krieg & Wade (2023) | 35 birds (17 male, 18 female) | No bird has ten calls on each of two recordings or dates, which the admission rule specified in advance requires. |
| Black-capped and Carolina chickadee hybrid zone, Palmer et al. (2025) | 55 genotyped birds; song recordings from 10 | Ten recorded birds are too few for a design across several days. |
| Ovenbird, Lapp et al. (2025) | Public code sample: 100 localised calls from 10 individuals; evaluation set described in the paper: 3,963 clips from 45 individuals | The public sample is too small, and the evaluation set named in the paper was not obtained. |"""


DONOR_LEVELS = (("-10db", "+10 dB"), ("0db", "0 dB"), ("10db", "−10 dB"))


def donor_table(donors: dict[str, Any]) -> str:
    rows = []
    for ep in BACKGROUND_ORDER:
        base = donors[ep]["none"]
        for key, word in DONOR_LEVELS:
            r = donors[ep][key]
            rows.append(
                [
                    NAMES[ep],
                    word,
                    r["wrong"],
                    f"{100 * base['named_donor']:.1f}%",
                    f"{100 * r['named_donor']:.1f}%",
                    f"{100 * r['reassigned']:.1f}%",
                    p4(r["p"]),
                ]
            )
    return (
        "**Table S7.** Exploratory analysis, specified after the results were seen: when "
        "BirdNET named the wrong individual for a clip mixed with another individual's "
        "background, the share of those errors that named the donor.\n\n"
        + table(
            [
                "Dataset",
                "Added background relative to test clip",
                "Wrong answers",
                "Named the donor, no background added",
                "Named the donor, background added",
                "Under reassigned donors",
                "p",
            ],
            rows,
        )
        + "\n\n*Notes:* Each individual's donor is the next individual in sorted order, so "
        "the donor is fixed for all of an individual's clips; a hash chooses only which of the "
        "donor's background recordings is used. No background added: the same share among the "
        "errors on the unmixed test clips, which shows how often BirdNET already confused each "
        "individual with its donor. Reassigned donors: 9,999 random reassignments of donors "
        "among the wrong answers, never giving a clip its own individual; p is the share at or "
        "above the observed value. This comparison does not control for the fixed pairing, so "
        "the column without background is the one to compare with."
    )


def full_name(block: dict[str, Any], ep: str | None, model: str) -> str:
    """The ledger key's model name, which keeps the publisher prefix for speech networks."""
    prefix = "" if ep is None else f"{ep}.controls."
    names = [k[len(prefix) :] for k in block if k.startswith(prefix)]
    found = [n for n in names if n == model or n.split("/")[-1] == model]
    if len(found) != 1:
        raise KeyError(f"{ep}: {model}")
    return found[0]


TREATMENT = {
    "none": "none",
    "per-session mean, batch": "session mean subtracted, whole session",
    "per-session mean, streaming": "session mean subtracted, running",
    "within-class covariance normalisation": "within-class covariance normalisation",
    "between-session subspace removed": "directions of between-session variation removed",
}


def setting(value: Any) -> str:
    return "n/a" if value is None else str(value)


def relabel_place(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines[2:], start=2):
        cells = line.strip("|").split("|")
        cells[-1] = f" {float(cells[-1]):.1f} "
        lines[i] = "|" + "|".join(cells) + "|"
    text = "\n".join(lines)
    text = text.replace(
        "| Bird | Same nest box both sides | Same recording both sides | Years "
        "differ | Metres between nest boxes |",
        "| Bird | Same nest box on both sides | Same recording on both sides | "
        "Years differ | Distance between nest boxes (m) |",
    )
    return text


def s3_row(name: str, h: dict[str, Any], ep: str) -> list[Any]:
    def opt(key: str) -> str:
        return "n/a" if h.get(key) is None or ep in NO_PAIRED_CLIP else f3(h[key])

    within = (
        "n/a"
        if h["within_95"] is None or ep in NO_SHARED_RECORDINGS
        else f"{f3(h['within_95'])} ({p4(h['within_p'])})"
    )
    three = (
        "n/a"
        if h["three_stage"] is None
        else f"{f3(h['three_stage'][0])} to {f3(h['three_stage'][1])}"
    )
    return [
        name,
        h["individuals"],
        f"{h['enrolment_clips']:,} / {h['query_clips']:,}",
        iv(h["accuracy"], h["interval"]),
        three,
        f3(h["uniform_chance"]),
        f3(h["majority"]),
        f3(h["permutation_95"]),
        p4(h["permutation_p"]),
        within,
        opt("trials_difference"),
        opt("trials_standardised"),
        opt("trials_eer"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    b = Build(args.build, args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables-main.md").write_text(main_tables(b))
    si = supporting_tables(b)
    (args.out / "tables-si.md").write_text("\n\n".join(si[k] for k in si) + "\n")


if __name__ == "__main__":
    main()
