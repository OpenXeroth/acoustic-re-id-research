"""Quantities added in paper v7 after the fourth review round, from built ledgers and results.

Run by scripts/paper_v6_build.sh after the evidence gate and the v6 renderers. It reads only
files whose SHA-256 the evidence report or the v6 comparison records, and writes one ledger,
OUT/review/review-numbers.json, plus Figure 7 without the duplicate WavLM Base+ entry.

1. The network comparison under three layer rules, with the speaker-verification WavLM Base+
   checkpoint left out because its weights and every value equal WavLM Base+: the Fisher rule
   and the held-out-session rule as registered, and each network's own final output
   (specified after the review, so exploratory). The held-out rule is summarised only on the
   datasets where it can choose among layers. Holm's correction runs over the 37 comparisons
   with BirdNET on each dataset and rule.
2. Counts over the 38 distinct entries for the background, added-background, score
   normalisation, within-recording and open-set tests, unadjusted and after Holm's correction.
3. Realised stranger acceptance and chance for the open-set rows of Table 6.
4. The fruit bat correlation over distinct session-compensation settings.
5. The donor-naming table with the rate before any background was added.
6. The right whale pairings with Holm's correction over the 11 frequency shifts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from paper_figures import Numbers
from paper_v6_controls import complete_gate
from paper_v6_evidence import BACKGROUND_ENDPOINTS, ENDPOINTS, OPEN_ENDPOINTS
from paper_v6_overview import DISPLAY, SHORT
from scipy.stats import spearmanr

from xinyenyana.encoder_comparison import paired_bootstrap, rank_consistency
from xinyenyana.v6_analyses import merge_sweeps

DUPLICATE = "wavlm-base-plus-sv"
REFERENCE = "birdnet-v2.4"
REPLICATES = 10000
PRIMARY = [e for e in ENDPOINTS if e != "great-tit-full"]
ENTRIES = [REFERENCE] + [m for m in DISPLAY if m != DUPLICATE]
DONOR_ARMS = (("other_-10db", "-10db"), ("other_0db", "0db"), ("other_10db", "10db"))
DONOR_FILES = {
    "chiffchaff-withinyear": "background-chiffchaff-withinyear.json",
    "pipit-withinyear": "background-pipit-withinyear.json",
    "pipit-acrossyear": "background-pipit-acrossyear.json",
    "chiffchaff-acrossyear": "background-chiffchaff-acrossyear.json",
    "littleowl-acrossyear": "background-littleowl.json",
}


def short(name: str) -> str:
    return name.split("/")[-1]


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda k: pvalues[k])
    count, running, out = len(ordered), 0.0, {}
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalues[key]))
        out[key] = running
    return out


def final_output(curve: dict[str, Any]) -> str:
    """The candidate that is the network's own output: its embedding, or last layer at speed 1."""
    if "embedding" in curve:
        return "embedding"
    single = [k for k, e in curve.items() if e.get("layer") is None and e.get("slowdown") == 1]
    if len(single) == 1:
        return single[0]
    layered = [k for k, e in curve.items() if e.get("layer") is not None and e["slowdown"] == 1]
    return max(layered, key=lambda k: curve[k]["layer"])


def compare(merged: dict[str, Any], chosen: dict[str, Any]) -> dict[str, Any]:
    correct = {short(m): e["query_correct"] for m, e in chosen.items()}
    paired = paired_bootstrap(
        identities=merged["query_identities"], correct=correct, replicates=REPLICATES
    )
    adjusted = holm({m: c["p_two_sided"] for m, c in paired["comparisons"].items()})
    family = len(adjusted)
    for model, entry in paired["comparisons"].items():
        entry["p_holm"] = adjusted[model]
        entry["differs_at_0.05_after_holm"] = adjusted[model] < 0.05
    paired["holm_family"] = family
    # With no draw on the far side of zero the two-sided p is at most 2 / draws before
    # adjustment, and Holm's factor is at most the family size.
    paired["zero_p_bound"] = family * 2 / REPLICATES
    selection = {
        short(m): {"selected": e["representation"], "accuracy": e["accuracy"]}
        for m, e in chosen.items()
    }
    return {"selection": selection, "paired": paired}


def comparison(results: Path, numbers: Numbers) -> dict[str, Any]:
    v6 = numbers.load("v6-encoder-comparison.json")
    per: dict[str, Any] = {}
    accuracy: dict[str, dict[str, dict[str, float]]] = {"Fisher": {}, "held_out": {}, "final": {}}
    changes = {"changed": 0, "higher": 0, "lower": 0, "same": 0, "deltas": []}
    for ep in PRIMARY:
        loaded = []
        for path, digest in v6["per_endpoint"][ep]["sources"].items():
            name = Path(path).name
            raw = (results / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(f"{name}: bytes differ from the v6 comparison's record")
            numbers.sources[name] = digest
            loaded.append(json.loads(raw))
        merged = merge_sweeps(loaded)
        curve = {m: c for m, c in merged["curve"].items() if short(m) != DUPLICATE}
        fisher = {m: e for m, e in merged["chosen"].items() if short(m) != DUPLICATE}
        held = {m: e for m, e in merged["chosen_held_out"].items() if short(m) != DUPLICATE}
        applies = any(len(curve[m]) > 1 for m in held)
        final = {}
        for model in fisher:
            key = final_output(curve[model])
            final[model] = {**curve[model][key], "representation": key}
        row = {"Fisher": compare(merged, fisher), "final": compare(merged, final)}
        # Fisher-rule values must equal the registered comparison, draw for draw.
        old = v6["per_endpoint"][ep]["paired"]["comparisons"]
        for model, entry in row["Fisher"]["paired"]["comparisons"].items():
            o = old[model]
            if not (
                math.isclose(entry["accuracy"], o["accuracy"], abs_tol=1e-12)
                and math.isclose(
                    entry["difference_from_reference"],
                    o["difference_from_reference"],
                    abs_tol=1e-12,
                )
                and np.allclose(entry["difference_95"], o["difference_95"], atol=1e-12)
            ):
                raise ValueError(f"{ep} {model}: Fisher comparison does not reproduce v6")
        if applies:
            row["held_out"] = compare(merged, held)
            for model in held:
                a, b = fisher[model], held[model]
                if a["representation"] != b["representation"]:
                    changes["changed"] += 1
                    d = b["accuracy"] - a["accuracy"]
                    changes["deltas"].append(d)
                    changes["higher" if d > 0 else "lower" if d < 0 else "same"] += 1
        row["held_out_applies"] = applies
        per[ep] = row
        for rule in ("Fisher", "final") + (("held_out",) if applies else ()):
            accuracy[rule][ep] = {m: s["accuracy"] for m, s in row[rule]["selection"].items()}
    counts = {}
    for rule in ("Fisher", "held_out", "final"):
        above = below = total = 0
        for row in per.values():
            if rule not in row:
                continue
            for entry in row[rule]["paired"]["comparisons"].values():
                total += 1
                if entry["differs_at_0.05_after_holm"]:
                    above += entry["difference_from_reference"] > 0
                    below += entry["difference_from_reference"] < 0
        counts[rule] = {"comparisons": total, "above": above, "below": below}
    ranks = {rule: rank_consistency(acc) for rule, acc in accuracy.items()}
    deltas = changes.pop("deltas")
    changes["median_change"] = statistics.median(deltas) if deltas else None
    changes["comparisons"] = sum(
        len(row["held_out"]["paired"]["comparisons"]) for row in per.values() if "held_out" in row
    )
    groups: dict[str, list[float]] = {}
    for row in per.values():
        for model, s in row["final"]["selection"].items():
            f = row["Fisher"]["selection"][model]
            group = group_of(model)
            if group is None:
                continue
            groups.setdefault(group, []).append(s["accuracy"] - f["accuracy"])
    final_vs_fisher = {
        g: {
            "pairs": len(v),
            "final_higher": sum(d > 0 for d in v),
            "final_lower": sum(d < 0 for d in v),
            "equal": sum(d == 0 for d in v),
            "median": statistics.median(v),
        }
        for g, v in groups.items()
    }
    return {
        "per_endpoint": per,
        "counts": counts,
        "ranks": ranks,
        "held_out_changes": changes,
        "final_vs_fisher": final_vs_fisher,
    }


ANIMAL_MULTI = {
    "esp-aves2-sl-beats-all",
    "avesecho-passt",
    "birdmae",
    "protoclr",
    "birdaves",
    "aves",
    "naturebeats",
    "biolingual",
    "beats",
    "audiomae",
}
SPEAKER = {"spkrec-ecapa-voxceleb", "spkrec-resnet-voxceleb", "spkrec-xvect-voxceleb"}
SPEECH_MULTI = {
    "wav2vec2-base",
    "wav2vec2-large-robust",
    "wav2vec2-xls-r-300m",
    "wav2vec2-conformer-rope-large",
    "mms-300m",
    "hubert-base-ls960",
    "hubert-large-ll60k",
    "data2vec-audio-base-100h",
    "data2vec-audio-base-960h",
    "wavlm-base-plus",
    "wavlm-large",
    "unispeech-sat-base-plus",
    "xeus",
}


def group_of(model: str) -> str | None:
    if model in ANIMAL_MULTI:
        return "animal networks with several layers"
    if model in SPEECH_MULTI:
        return "speech networks with several layers"
    if model in SPEAKER:
        return "speaker networks"
    return None


def figure_seven(per: dict[str, Any], rule: str, out: Path) -> None:
    """Figure 7: every distinct entry against BirdNET on the 13 datasets."""
    from matplotlib.colors import LinearSegmentedColormap
    from paper_figures import BLUE, ORANGE, plt

    cmap = LinearSegmentedColormap.from_list("difference", [ORANGE, "white", BLUE])
    cmap.set_bad("#dddddd")
    columns = [e for e in SHORT if e in per and rule in per[e]]
    models = ENTRIES[1:]
    matrix = np.full((len(models), len(columns)), np.nan)
    marks = []
    for x, ep in enumerate(columns):
        comps = per[ep][rule]["paired"]["comparisons"]
        for y, model in enumerate(models):
            entry = comps.get(model)
            if entry is not None:
                matrix[y, x] = entry["difference_from_reference"]
                if entry["differs_at_0.05_after_holm"]:
                    marks.append((x, y))
    fig, axis = plt.subplots(figsize=(9.6, 8.3))
    picture = axis.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    if marks:
        xs, ys = zip(*marks, strict=True)
        axis.scatter(xs, ys, marker=".", color="black", s=16)
    axis.set(
        xticks=range(len(columns)),
        xticklabels=[SHORT[e].replace("\n", " ") for e in columns],
        yticks=range(len(models)),
        yticklabels=[DISPLAY[m] for m in models],
    )
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", labelsize=8, rotation=40)
    plt.setp(axis.get_xticklabels(), ha="left", rotation_mode="anchor")
    axis.tick_params(axis="y", labelsize=8)
    speech_start = models.index("spkrec-ecapa-voxceleb")
    control_start = models.index("clip-level-only")
    for boundary in (speech_start - 0.5, control_start - 0.5):
        axis.axhline(boundary, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(columns) - 1) + 0.5, minor=True)
    axis.set_yticks(np.arange(len(models) - 1) + 0.5, minor=True)
    axis.grid(which="minor", color="#bbbbbb", linewidth=0.2)
    axis.tick_params(which="minor", length=0)
    bar = fig.colorbar(picture, ax=axis, shrink=0.5, pad=0.02)
    bar.set_label("Re-ID accuracy minus BirdNET's", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def ledger(build: Path, relative: str) -> dict[str, Any]:
    data = json.loads((build / relative).read_text())
    return {k: v["value"] for k, v in data["values"].items()}


def keyed(block: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Map short model names to ledger entries under ``prefix``."""
    out = {}
    for key, value in block.items():
        if key.startswith(prefix):
            out[short(key[len(prefix) :])] = value
    return out


def span(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def entry_counts(build: Path, results: Path) -> dict[str, Any]:
    controls = ledger(build, "recording-controls/recording-controls-numbers.json")
    out: dict[str, Any] = {"entries": len(ENTRIES)}
    background = {}
    for ep in BACKGROUND_ENDPOINTS:
        models = keyed(controls, f"{ep}.controls.")
        models = {m: v for m, v in models.items() if m in ENTRIES}
        if set(models) != set(ENTRIES):
            raise ValueError(f"{ep}: background controls do not cover the 38 entries")
        pairs: dict[str, Any] = {}
        for pairing in (
            "background_gallery_background_query",
            "foreground_gallery_background_query",
            "background_gallery_foreground_query",
            "foreground_gallery_foreground_query",
        ):
            p = {m: v["four_pairings"][pairing]["permutation_p"] for m, v in models.items()}
            adj = holm(p)
            pairs[pairing] = {
                "unadjusted_below_0.05": sum(v < 0.05 for v in p.values()),
                "holm_below_0.05": sum(v < 0.05 for v in adj.values()),
                "holm_passing": sorted(m for m, v in adj.items() if v < 0.05),
                "unadjusted_failing": sorted(m for m, v in p.items() if v >= 0.05),
            }
        challenge = {}
        for level in ("-10db", "0db", "10db"):
            ci = {
                m: v["challenge"]["own_vs_other"][level]["identity_block_difference_95"]
                for m, v in models.items()
            }
            challenge[level] = {
                "positive_interval": sum(c[0] > 0 for c in ci.values()),
                "negative_interval": sum(c[1] < 0 for c in ci.values()),
            }
        norms = {short(k): v for k, v in controls[f"{ep}.as_norm"].items()}
        norms = {m: v for m, v in norms.items() if m in ENTRIES}
        as_norm = {
            "calls_lower": sum(
                v["calls_scored"]["as_norm"]["accuracy"]
                < v["calls_scored"]["raw_class_mean"]["accuracy"]
                for v in norms.values()
            ),
            "backgrounds_higher": sum(
                v["backgrounds_scored"]["as_norm"]["accuracy"]
                > v["backgrounds_scored"]["raw_class_mean"]["accuracy"]
                for v in norms.values()
            ),
            "entries": len(norms),
        }
        background[ep] = {"pairings": pairs, "challenge": challenge, "as_norm": as_norm}
    out["background"] = background
    within = {}
    for ep in (
        "birdpark-juv01",
        "birdpark-juv03",
        "rookid-full-width",
        "zebra-finch",
        "bat-acrosstreatment",
    ):
        info = {
            short(k): v
            for k, v in ledger(build, f"information/{ep}-numbers.json")["models"].items()
        }
        analyses = json.loads((results / f"v6-analyses-{ep}.json").read_text())["models"]
        rows = {}
        for m in ENTRIES:
            f = info[m]["fisher"]
            a = [v for k, v in analyses.items() if short(k) == m][0]
            null = a.get("permutation_nulls", {}).get("within_session_permutation") or {}
            rows[m] = {
                "accuracy": f["accuracy"],
                "within_recording_p": f["within_recording_p"],
                "within_recording_95": null.get("percentile_95"),
            }
        within[ep] = {
            "rows": rows,
            "below_0.05": sum(
                r["within_recording_p"] is not None and r["within_recording_p"] < 0.05
                for r in rows.values()
            ),
        }
    out["within_recording"] = within
    return out


def calibrated_against_test_derived(results: Path, sources: dict[str, str]) -> dict[str, int]:
    """At the 10% budget, count allocations where the calibrated threshold named more known
    birds than the test-derived one, and whether it had then accepted more strangers."""
    counts = {"allocations": 0, "over_budget": 0, "above_test_derived": 0}
    counts["above_test_derived_within_budget"] = 0
    for ep in OPEN_ENDPOINTS:
        name = f"v6-counts/v6-open-set-{ep}.json"
        raw = (results / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sources[name]:
            raise ValueError(f"{name}: bytes differ from the evidence report")
        data = json.loads(raw)
        for name, row in data["representations"].items():
            if short(name) not in ENTRIES:
                continue
            for allocation in row["allocations"]:
                test = allocation["calibrated_on_other_birds"]["0.10"]["test"]
                over = test["unknown_false_accept_rate"] > 0.10
                above = (
                    test["known_correct_accept_rate"]
                    > allocation["best_acceptance_at_budget"]["0.10"]
                )
                counts["allocations"] += 1
                counts["over_budget"] += over
                counts["above_test_derived"] += above
                counts["above_test_derived_within_budget"] += above and not over
    return counts


def open_set(build: Path, v5: Path, results: Path, sources: dict[str, str]) -> dict[str, Any]:
    opened = ledger(build, "open-set/open-set-numbers.json")
    roles = json.loads((build / "open-set/open-set-role-counts.json").read_text())["endpoints"]
    out: dict[str, Any] = {}
    for ep, entry in opened.items():
        models = {short(k): v["allocation_sensitivity"] for k, v in entry["models"].items()}
        known = [a["test_known"] for a in roles[ep]["allocations"]]
        out[ep] = {
            "chance": {"minimum": 1 / max(known), "maximum": 1 / min(known)},
            "test_known": {"minimum": min(known), "maximum": max(known)},
            "leader": short(entry["models_by_rule"]["best average rank"]),
            "achieved": {
                m: models[m]["achieved_stranger_acceptance_calibrated_at_0.10"] for m in ENTRIES
            },
            "maximum_accepted_and_named": max(
                models[m]["acceptance_calibrated_at_0.10"]["median"] for m in ENTRIES
            ),
            "best": max(
                ENTRIES, key=lambda m: models[m]["acceptance_calibrated_at_0.10"]["median"]
            ),
        }
    data = json.loads((v5 / "v5-open-set-great-tit.json").read_text())
    for rep in ("masked-birdnet-v2.4", "acquisition-context"):
        rates = [
            a["calibrated_on_other_birds"]["0.10"]["test"]["unknown_false_accept_rate"]
            for a in data["representations"][rep]["allocations"]
        ]
        out["great-tit"].setdefault("v5_achieved", {})[rep] = span(rates)
    out["calibrated_against_test_derived"] = calibrated_against_test_derived(results, sources)
    return out


def bat(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    settings = []
    for treatment, rows in data["treatments"].items():
        for row in rows:
            if row["head"].startswith("kernel"):
                settings.append(
                    (
                        treatment,
                        row["parameter"],
                        row["accuracy"],
                        row["recording_day_probe"]["accuracy"],
                    )
                )
    none = [s for s in settings if s[0] == "none"][0]
    duplicates = [s for s in settings if s[0] != "none" and s[2] == none[2] and s[3] == none[3]]
    distinct = [s for s in settings if s not in duplicates]
    kept = [s for s in distinct if not s[0].startswith("per-session")]
    rho = spearmanr([s[2] for s in distinct], [s[3] for s in distinct])
    rho_kept = spearmanr([s[2] for s in kept], [s[3] for s in kept])
    return {
        "settings": len(settings),
        "identical_to_none": [[s[0], s[1]] for s in duplicates],
        "distinct": len(distinct),
        "spearman_distinct": [float(rho.statistic), float(rho.pvalue)],
        "distinct_without_mean_subtraction": len(kept),
        "spearman_without_mean_subtraction": [float(rho_kept.statistic), float(rho_kept.pvalue)],
    }


def donors(stage: Path) -> dict[str, Any]:
    from donor_direction import REPLICATES as DRAWS
    from donor_direction import SEED, donor_identity, donor_share

    out = {}
    for ep, name in DONOR_FILES.items():
        result = json.loads((stage / name).read_text())
        block = result["representations"][REFERENCE]
        identities = frozenset(block["arms"]["original"]["identity_accuracy"])
        pairs = {Path(p["foreground"]).name: p["donors"] for p in result["donor_pairs"]}
        rows = {}
        for arm, level in (("original", "none"),) + DONOR_ARMS:
            preds = block["arms"][arm]["predictions"]
            actual = np.array([r["actual_label"] for r in preds])
            predicted = np.array([r["predicted_label"] for r in preds])
            donor = np.array(
                [
                    donor_identity(pairs[r["query_id"]]["other"]["filename"], identities)
                    for r in preds
                ]
            )
            wrong, observed, shuffled, p = donor_share(
                actual, predicted, donor, replicates=DRAWS, seed=SEED
            )
            rows[level] = {"wrong": wrong, "named_donor": observed, "reassigned": shuffled, "p": p}
        out[ep] = rows
    return out


def whale(results: Path) -> dict[str, Any]:
    shifts = list(range(0, 10001, 1000))
    p: dict[str, dict[int, float]] = {}
    for shift in shifts:
        data = json.loads(
            (results / "carried" / f"gate-right-whale-shift-{shift}.json").read_text()
        )
        for pairing, value in data["pairings"].items():
            p.setdefault(pairing, {})[shift] = value["permutation_p"]
    out = {}
    for pairing, values in p.items():
        adj = holm({str(k): v for k, v in values.items()})
        out[pairing] = {
            "unadjusted_below_0.05": sum(v < 0.05 for v in values.values()),
            "holm_below_0.05": sum(v < 0.05 for v in adj.values()),
            "holm": adj,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--v5-results", type=Path, required=True)
    parser.add_argument("--bat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = complete_gate(args.build / "evidence.json")
    numbers = Numbers(args.results)
    comp = comparison(args.results, numbers)
    if (
        numbers.sources["v6-encoder-comparison.json"]
        != report["sources"]["v6-encoder-comparison.json"]
    ):
        raise ValueError("comparison bytes differ from the evidence report")
    numbers.put("comparison", comp, "v6-encoder-comparison.json and the sweeps it names")
    numbers.put("counts", entry_counts(args.build, args.results), "build ledgers")
    numbers.put(
        "open_set",
        open_set(args.build, args.v5_results, args.results, report["sources"]),
        "open-set ledgers and v6-counts open-set files",
    )
    numbers.put("bat", bat(args.bat), args.bat.name)
    numbers.put("donors", donors(args.build / "carried-stage"), "carried-stage background files")
    numbers.put("whale", whale(args.results), "carried right whale gates")
    args.out.mkdir(parents=True, exist_ok=True)
    numbers.write(args.out / "review-numbers.json")
    figure_seven(comp["per_endpoint"], "Fisher", args.out / "fig7-comparison-v7.png")
    print(f"wrote {args.out / 'review-numbers.json'}")


if __name__ == "__main__":
    main()
