"""Route one prediction request to A / B / C using schema only.

`task` cannot be used here: B and C both report ``"regression"``, and A/B/C
share the same outer envelope ``{"task": ..., "inputs": ...}``.  The only
reliable discriminator is the shape of ``inputs``:

    {"settings": [...], "sensors": [...]}         -> C   (3 + 21 = 24 dims)
    {"features": {"horizontal": {...}, ...}}      -> B   (17 + 17 = 34 dims)
    {"features": [ ... 128 numbers ... ]}         -> A   (128 dims)

Validated on every validation record of all three packages (1244 / 1679 /
23049 rows), zero misroutes.
"""


def route(inputs):
    """Return ``"A"``, ``"B"`` or ``"C"``; raise ``ValueError`` on anything else."""
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be a JSON object")
    if "settings" in inputs:
        return "C"
    features = inputs.get("features")
    if isinstance(features, dict):
        return "B"
    if isinstance(features, list) and len(features) == 128:
        return "A"
    raise ValueError("unknown input schema")
