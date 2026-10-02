"""Where the wrong answers go in a background challenge's other-donor arms.

Unregistered exploratory analysis, specified after the five challenge results had
been read. It refits nothing and reads no audio: it uses only the predictions and
donor pairings that `run-background-challenge` already retains in its result.

For each query the other-donor arm answered wrongly, it asks whether the answer
was the donor's identity. The comparison shuffles the donor identities across the
wrong-answered queries, holding the pool of donors fixed and forbidding a query
its own identity.

    python scripts/donor_direction.py background-chiffchaff-withinyear.json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPRESENTATION = "birdnet-v2.4"
ARMS = ("other_-10db", "other_0db", "other_10db")
REPLICATES = 9999
SEED = 17


def donor_identity(filename: str, identities: frozenset[str]) -> str:
    """The identity token of a donor clip, checked rather than guessed.

    Donor filenames are `<corpus and split>_<identity>_<index>.wav`, so the
    identity is the second-to-last underscore token. A token that is not one of
    the endpoint's identities raises instead of being interpreted.
    """
    token = Path(filename).name.rsplit(".", 1)[0].split("_")[-2]
    if token not in identities:
        raise ValueError(f"token {token!r} from {filename!r} is not an identity")
    return token


def donor_share(
    actual: np.ndarray, predicted: np.ndarray, donor: np.ndarray, *, replicates: int, seed: int
) -> tuple[int, float, float, float]:
    """Observed share of wrong answers naming the donor, the shuffled share, and p."""
    if bool((donor == actual).any()):
        raise ValueError("a donor carries the query's own identity")
    wrong = predicted != actual
    if not wrong.any():
        raise ValueError("no wrong answers to read")
    true_labels, answers, pool = actual[wrong], predicted[wrong], donor[wrong]
    observed = float(np.mean(answers == pool))
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        shuffled = generator.permutation(pool)
        collides = shuffled == true_labels
        while collides.any():
            shuffled[collides] = generator.permutation(pool)[collides]
            collides = shuffled == true_labels
        draws[index] = float(np.mean(answers == shuffled))
    p_value = (int((draws >= observed).sum()) + 1) / (replicates + 1)
    return int(wrong.sum()), observed, float(draws.mean()), p_value


def read(path: Path, *, replicates: int, seed: int) -> None:
    result = json.loads(path.read_text())
    block = result["representations"][REPRESENTATION]
    identities = frozenset(block["arms"]["original"]["identity_accuracy"])
    donors = {Path(pair["foreground"]).name: pair["donors"] for pair in result["donor_pairs"]}
    print(f"=== {result['endpoint']}  identities={len(identities)} queries={len(donors)}")
    for arm in ARMS:
        predictions = block["arms"][arm]["predictions"]
        actual = np.array([row["actual_label"] for row in predictions])
        predicted = np.array([row["predicted_label"] for row in predictions])
        donor = np.array(
            [
                donor_identity(donors[row["query_id"]]["other"]["filename"], identities)
                for row in predictions
            ]
        )
        wrong, observed, shuffled, p_value = donor_share(
            actual, predicted, donor, replicates=replicates, seed=seed
        )
        print(
            f"  {arm:<12} wrong={wrong:>4}  answered the donor {100 * observed:5.1f}%"
            f"  shuffled donors {100 * shuffled:5.1f}%  p={p_value:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="background challenge result files")
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    for path in arguments.results:
        read(path, replicates=arguments.replicates, seed=arguments.seed)


if __name__ == "__main__":
    main()
