"""Compare completed endpoint results on one fixed colour scale, without pooling tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from paper_figures import BLUE, ORANGE, md_table, plt
from paper_v6_evidence import ALL_MODELS, ENDPOINTS

LABELS = {
    "birdpark-juv01": "Zebra finch\ngroup of four",
    "birdpark-juv03": "Zebra finch\ngroup of eight",
    "zebra-finch": "Zebra finch\nseparate recordings",
    "great-tit": "Great tit\nacross year\nnest-attributed",
    "pipit-acrossyear": "Tree pipit\nacross year",
    "pipit-withinyear": "Tree pipit\nwithin year",
    "chiffchaff-acrossyear": "Chiffchaff\nacross year",
    "chiffchaff-withinyear": "Chiffchaff\nwithin year",
    "littleowl-acrossyear": "Little owl\nacross year",
    "penguin-acrossnight": "Little penguin\nacross night",
    "cockatoo-fold1": "Cockatoo\nrandom fold",
    "bat-acrosstreatment": "Egyptian fruit bat\nacross year",
    "rookid-full-width": "Rook\nfull aviary",
}


#: Names used in the paper, grouped as in Table S9: networks trained on animal or general
#: sound, speech networks, then the two controls.
DISPLAY = {
    "perch-v2": "Perch 2.0",
    "perch-bird": "Perch",
    "surfperch": "SurfPerch",
    "birdnet-v3-preview": "BirdNET v3 (preview)",
    "esp-aves2-sl-beats-all": "AVEX sl-BEATs",
    "esp-aves2-effnetb0-all": "AVEX EfficientNet-B0",
    "avesecho-passt": "AvesEcho",
    "audioprotopnet": "AudioProtoPNet",
    "convnext-birdset": "ConvNeXt (BirdSet)",
    "birdmae": "Bird-MAE",
    "protoclr": "ProtoCLR",
    "rcl-fs-bsed": "RCL_FS_BSED",
    "birdaves": "BirdAVES",
    "aves": "AVES",
    "naturebeats": "NatureBEATs",
    "biolingual": "BioLingual",
    "beats": "BEATs",
    "audiomae": "AudioMAE",
    "vggish": "VGGish",
    "spkrec-ecapa-voxceleb": "ECAPA-TDNN (speaker)",
    "spkrec-resnet-voxceleb": "ResNet (speaker)",
    "spkrec-xvect-voxceleb": "x-vector (speaker)",
    "wav2vec2-base": "wav2vec 2.0 Base",
    "wav2vec2-large-robust": "wav2vec 2.0 Large (robust)",
    "wav2vec2-xls-r-300m": "XLS-R 300M",
    "wav2vec2-conformer-rope-large": "wav2vec 2.0 Conformer Large",
    "mms-300m": "MMS 300M",
    "hubert-base-ls960": "HuBERT Base",
    "hubert-large-ll60k": "HuBERT Large",
    "data2vec-audio-base-100h": "data2vec Base (100 h)",
    "data2vec-audio-base-960h": "data2vec Base (960 h)",
    "wavlm-base-plus": "WavLM Base+",
    "wavlm-base-plus-sv": "WavLM Base+ (speaker verification)",
    "wavlm-large": "WavLM Large",
    "unispeech-sat-base-plus": "UniSpeech-SAT Base+",
    "xeus": "XEUS",
    "clip-level-only": "Loudness control",
    "clip-duration-only": "Duration control",
}

#: Column labels for the paper's Figure 7.
SHORT = {
    "birdpark-juv01": "Zebra finch,\ngroup of four",
    "birdpark-juv03": "Zebra finch,\ngroup of eight",
    "rookid-full-width": "Rook",
    "zebra-finch": "Zebra finch,\none per recording",
    "chiffchaff-withinyear": "Chiffchaff,\nwithin year",
    "chiffchaff-acrossyear": "Chiffchaff,\nacross year",
    "littleowl-acrossyear": "Little owl",
    "pipit-withinyear": "Tree pipit,\nwithin year",
    "pipit-acrossyear": "Tree pipit,\nacross year",
    "great-tit": "Great tit",
    "penguin-acrossnight": "Little penguin",
    "cockatoo-fold1": "Cockatoo,\nrandom split",
    "bat-acrosstreatment": "Fruit bat",
}


def load_comparisons(paths: list[Path]) -> tuple[dict[str, Any], dict[str, str]]:
    rows: dict[str, Any] = {}
    sources = {}
    expected = {model.split("/")[-1] for model in ALL_MODELS}
    for path in paths:
        if path.name in sources:
            raise ValueError(f"repeated source filename: {path.name}")
        raw = path.read_bytes()
        data = json.loads(raw)
        sources[path.name] = hashlib.sha256(raw).hexdigest()
        for endpoint, row in data["per_endpoint"].items():
            if endpoint not in ENDPOINTS[:-1] or endpoint in rows:
                raise ValueError(f"non-primary or repeated endpoint: {endpoint}")
            rules = {"Fisher": row, "Session holdout": row["held_out_rule"]}
            clean = {}
            for rule, block in rules.items():
                names = set(block["selection"])
                if (rule == "Fisher" and names != expected) or not names <= expected:
                    raise ValueError(f"unexpected model coverage: {endpoint}, {rule}")
                paired = block["paired"]
                if paired["reference"] != "birdnet-v2.4" or paired["replicates"] != 10000:
                    raise ValueError("unexpected reference or bootstrap count")
                if set(paired["comparisons"]) != names - {"birdnet-v2.4"}:
                    raise ValueError("missing paired comparison")
                entries = {}
                for model, entry in paired["comparisons"].items():
                    delta = entry["difference_from_reference"]
                    if (
                        not math.isclose(
                            delta, entry["accuracy"] - paired["reference_accuracy"], abs_tol=1e-12
                        )
                        or entry["accuracy"] != block["selection"][model]["selected_accuracy"]
                    ):
                        raise ValueError("inconsistent comparison values")
                    entries[model] = {
                        key: entry[key]
                        for key in (
                            "accuracy",
                            "difference_from_reference",
                            "difference_95",
                            "p_holm",
                            "differs_at_0.05_after_holm",
                        )
                    }
                clean[rule] = {
                    "reference_accuracy": paired["reference_accuracy"],
                    "comparisons": entries,
                }
            rows[endpoint] = clean
    if not rows:
        raise ValueError("no completed endpoints")
    return rows, sources


def render(paths: list[Path], out: Path, *, partial: bool) -> None:
    rows, sources = load_comparisons(paths)
    if not partial and set(rows) != set(ENDPOINTS[:-1]):
        raise ValueError("all thirteen primary endpoints are required; use --partial explicitly")
    endpoints = [name for name in ENDPOINTS[:-1] if name in rows]
    models = sorted({model.split("/")[-1] for model in ALL_MODELS} - {"birdnet-v2.4"})
    out.mkdir(parents=True, exist_ok=True)
    scope = f"{len(endpoints)} of 13 primary endpoints"
    caption = (
        "Colour shows observed accuracy minus BirdNET accuracy, on a fixed −1 to +1 scale. "
        "A black dot marks a contrast passing the endpoint-and-rule-specific Holm gate. "
        "Grey cells marked × are unavailable. Unmarked coloured cells do not establish "
        "equivalence. These are separate endpoint tests; there is no pooled test or "
        "correction across endpoints or across both selection rules. The supplementary "
        "expanded great-tit endpoint is excluded.\n\n"
    )
    text = f"# Revision 6 comparison overview: {scope}\n\n"
    if partial:
        text += (
            "This is an explicitly partial overview of completed endpoints. It does not "
            "report the final across-endpoint rank test or establish paper completion.\n\n"
        )
    text += caption
    references = []
    cmap = LinearSegmentedColormap.from_list("accuracy_difference", [ORANGE, "white", BLUE])
    cmap.set_bad("#dddddd")
    for rule, stem in [("Fisher", "fisher"), ("Session holdout", "session-holdout")]:
        matrix = np.full((len(models), len(endpoints)), np.nan)
        significant = []
        missing = []
        for x, endpoint in enumerate(endpoints):
            block = rows[endpoint][rule]
            references.append(
                [LABELS[endpoint].replace("\n", ", "), rule, f"{block['reference_accuracy']:.3f}"]
            )
            for y, model in enumerate(models):
                entry = block["comparisons"].get(model)
                if entry is None:
                    missing.append((x, y))
                else:
                    matrix[y, x] = entry["difference_from_reference"]
                    if entry["differs_at_0.05_after_holm"]:
                        significant.append((x, y))
        # Five endpoint columns fit an A4 portrait figure without shrinking model
        # labels to the illegible size of a single thirteen-column chart.
        panels = [endpoints[start : start + 5] for start in range(0, len(endpoints), 5)]
        for panel_index, panel in enumerate(panels):
            start = panel_index * 5
            stop = start + len(panel)
            fig, axis = plt.subplots(figsize=(7.1, 9.2))
            picture = axis.imshow(matrix[:, start:stop], cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            for points, marker, colour in [
                (significant, ".", "black"),
                (missing, "x", "#777777"),
            ]:
                visible = [(x - start, y) for x, y in points if start <= x < stop]
                if visible:
                    xs, ys = zip(*visible, strict=True)
                    axis.scatter(xs, ys, marker=marker, color=colour, s=14, linewidths=0.6)
            panel_scope = f"; panel {panel_index + 1}/{len(panels)}" if len(panels) > 1 else ""
            axis.set(
                xticks=range(len(panel)),
                xticklabels=[LABELS[e] for e in panel],
                yticks=range(len(models)),
                yticklabels=models,
                title=f"{rule}: {scope}{panel_scope}\n• passes Holm gate; × unavailable",
            )
            axis.tick_params(axis="y", labelsize=8.5)
            axis.tick_params(axis="x", labelsize=8.5, rotation=35)
            plt.setp(axis.get_xticklabels(), ha="right", rotation_mode="anchor")
            axis.set_xticks(np.arange(len(panel) - 1) + 0.5, minor=True)
            axis.set_yticks(np.arange(len(models) - 1) + 0.5, minor=True)
            axis.grid(which="minor", color="#bbbbbb", linewidth=0.2)
            axis.tick_params(which="minor", length=0)
            bar = fig.colorbar(picture, ax=axis, shrink=0.65)
            bar.set_label("Accuracy difference from BirdNET", fontsize=9)
            bar.ax.tick_params(labelsize=8.5)
            fig.tight_layout()
            suffix = f"-{panel_index + 1}" if len(panels) > 1 else ""
            filename = f"comparison-{stem}{suffix}.png"
            fig.savefig(out / filename)
            plt.close(fig)
            text += f"![{rule} differences from BirdNET{panel_scope}]({filename})\n\n"
    text += md_table(["Endpoint", "Rule", "BirdNET accuracy"], references)
    text += (
        "\n\nThe [numerical ledger](comparison-overview-numbers.json) preserves every plotted "
        "difference, paired interval, adjusted tail proportion and source digest. "
        "The endpoint-specific tables give the selected representations.\n"
    )
    (out / "comparison-overview.md").write_text(text)
    (out / "comparison-overview-numbers.json").write_text(
        json.dumps({"sources": sources, "partial": partial, "endpoints": rows}, indent=1) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    render(args.comparison, args.out, partial=args.partial)


if __name__ == "__main__":
    main()
