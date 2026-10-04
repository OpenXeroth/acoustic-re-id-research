"""Scope of cosine-only analyses after the preprint control audit."""

EXCLUDED_CONTROLS = frozenset({"clip-duration-only", "clip-level-only"})
EXCLUSION_REASON = (
    "Raw duration and level controls are excluded: cosine unit normalisation "
    "removes magnitude (positive duration becomes constant). Their ridge results remain valid."
)


def is_control(model: str) -> bool:
    return model.split("/")[-1] in EXCLUDED_CONTROLS
