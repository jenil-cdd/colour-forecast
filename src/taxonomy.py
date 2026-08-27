"""Colour taxonomy for duvet SKUs.

The mapper's own ``color_family`` is missing for 114 of 193 ASINs — including
every 2026 launch — so it cannot be used as a modelling feature. This module
derives a complete, auditable taxonomy from the colour string instead.

Three kinds of features come out of it:

1. ``shade_family``  — the demand pool a colour competes in (the unit that the
   cannibalisation / family-depth mechanic operates on).
2. ``pattern_type``  — solid vs pattern vs stripe vs textured vs two-tone,
   which drives both velocity and return rate.
3. CIE Lab coordinates — a *continuous* colour space, so look-alike matching
   and tree splits can use real perceptual distance rather than one-hot labels.
   ``dist_to_white`` in Lab space directly encodes the "near-whites sell into
   the white halo" mechanic instead of hard-coding a near-white list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Colour anchors: approximate sRGB for each colour word used in the catalogue.
# Values are eyeballed to the swatch, not sampled from imagery; they only need
# to be ordinally sensible in lightness/chroma/hue for the features to work.
# ---------------------------------------------------------------------------
COLOUR_RGB: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "antique white": (250, 243, 230),
    "ivory": (250, 246, 232),
    "cream": (250, 244, 227),
    "vanilla yellow": (247, 238, 197),
    "silver": (204, 206, 209),
    "silver gray": (195, 198, 201),
    "light gray": (200, 203, 207),
    "light grey": (200, 203, 207),
    "gray": (140, 144, 150),
    "grey": (140, 144, 150),
    "dark gray": (110, 114, 120),
    "slate gray": (108, 118, 130),
    "charcoal gray": (78, 82, 88),
    "black": (32, 32, 34),
    "taupe": (176, 160, 145),
    "warm taupe": (170, 152, 136),
    "beige": (214, 199, 176),
    "silver sage": (176, 186, 176),
    "sage green": (156, 175, 152),
    "olive green": (124, 128, 82),
    "seafoam": (160, 205, 190),
    "sky blue": (150, 190, 220),
    "blue hydrangea": (140, 165, 205),
    "indigo dusty blue": (110, 135, 165),
    "navy": (42, 56, 96),
    "indigo navy blue": (40, 54, 92),
    "lavender grey": (180, 176, 190),
    "pink": (232, 190, 196),
    "burgundy red": (110, 36, 46),
    "blue": (110, 140, 190),
    # Colourless pattern/texture descriptors. The mapper's own color_family is
    # used as the cross-check where it exists: "Natural Linen Look Slub" is
    # filed under Ivory, "Polka Dot" under Navy Dot, and CDD pinstripe is a
    # white ground with a fine grey stripe.
    "natural": (238, 230, 212),
    "pinstripe": (242, 242, 244),
    "polka dot": (236, 238, 242),
}

# Colour word -> demand-pool family. Ordering matters: longest match wins, so
# "indigo navy blue" resolves before "blue".
FAMILY_RULES: list[tuple[str, str]] = [
    ("antique white", "Near White"),
    ("white", "White"),
    ("ivory", "Near White"),
    ("cream", "Near White"),
    ("vanilla yellow", "Yellow"),
    ("silver sage", "Green"),
    ("silver gray", "Light Grey"),
    ("silver", "Light Grey"),
    ("light gray", "Light Grey"),
    ("light grey", "Light Grey"),
    ("lavender grey", "Purple Grey"),
    ("charcoal gray", "Dark Grey"),
    ("slate gray", "Dark Grey"),
    ("dark gray", "Dark Grey"),
    ("gray", "Mid Grey"),
    ("grey", "Mid Grey"),
    ("black", "Black"),
    ("warm taupe", "Taupe"),
    ("taupe", "Taupe"),
    ("beige", "Beige"),
    ("sage green", "Green"),
    ("olive green", "Green"),
    ("seafoam", "Green"),
    ("indigo navy blue", "Dark Blue"),
    ("navy", "Dark Blue"),
    ("indigo dusty blue", "Mid Blue"),
    ("blue hydrangea", "Mid Blue"),
    ("sky blue", "Light Blue"),
    ("blue", "Mid Blue"),
    ("burgundy red", "Red"),
    ("pink", "Pink"),
    ("natural", "Near White"),
    ("pinstripe", "White"),
    ("polka dot", "White"),
]

PATTERN_PREFIXES = {
    "pattern": "Pattern",
    "stripe": "Stripe",
    "textured": "Textured",
}


@dataclass(frozen=True)
class ColourAttrs:
    colour_clean: str
    shade_family: str
    pattern_type: str
    base_colour: str
    n_colour_words: int
    lab_L: float
    lab_a: float
    lab_b: float


def _srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """sRGB (0-255) -> CIE L*a*b* under D65."""
    r, g, b = (c / 255.0 for c in rgb)

    def inv_gamma(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = inv_gamma(r), inv_gamma(g), inv_gamma(b)
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    # D65 white point
    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


LAB_WHITE = _srgb_to_lab(COLOUR_RGB["white"])


def _match_longest(text: str, keys) -> str | None:
    """Longest-substring match so multi-word colours beat their components."""
    best = None
    for k in keys:
        if k in text and (best is None or len(k) > len(best)):
            best = k
    return best


def parse_colour(colour: str) -> ColourAttrs:
    raw = (colour or "").strip()
    low = raw.lower()

    # Pattern / texture prefix, e.g. "Textured - White Waffle"
    pattern_type = "Solid"
    body = low
    for prefix, label in PATTERN_PREFIXES.items():
        if low.startswith(prefix):
            pattern_type = label
            body = re.sub(rf"^{prefix}\s*-\s*", "", low)
            break

    # Two-tone colours use a slash, e.g. "White/Gray", "Ivory/Sage"
    parts = [p.strip() for p in body.split("/") if p.strip()]
    if len(parts) > 1 and pattern_type == "Solid":
        pattern_type = "Two Tone"

    primary = parts[0] if parts else body

    fam_key = _match_longest(primary, [k for k, _ in FAMILY_RULES])
    if fam_key is None:
        fam_key = _match_longest(body, [k for k, _ in FAMILY_RULES])
    shade_family = dict(FAMILY_RULES).get(fam_key, "Other") if fam_key else "Other"

    rgb_key = _match_longest(primary, COLOUR_RGB.keys()) or _match_longest(body, COLOUR_RGB.keys())
    rgb = COLOUR_RGB.get(rgb_key, (170, 170, 170))  # neutral mid-grey fallback
    L, a, b = _srgb_to_lab(rgb)

    return ColourAttrs(
        colour_clean=raw,
        shade_family=shade_family,
        pattern_type=pattern_type,
        base_colour=rgb_key or "unknown",
        n_colour_words=len(parts),
        lab_L=L,
        lab_a=a,
        lab_b=b,
    )


def annotate(sku_dim: pd.DataFrame, colour_col: str = "colour") -> pd.DataFrame:
    """Attach taxonomy columns to a SKU dimension table."""
    attrs = [parse_colour(c) for c in sku_dim[colour_col].fillna("")]
    extra = pd.DataFrame([a.__dict__ for a in attrs], index=sku_dim.index)

    out = pd.concat([sku_dim, extra.drop(columns=["colour_clean"])], axis=1)
    out["is_core_white"] = (out["shade_family"] == "White") & (out["pattern_type"] == "Solid")
    out["is_near_white"] = out["shade_family"].isin(["White", "Near White"])
    out["is_solid"] = out["pattern_type"] == "Solid"

    # Perceptual distance to pure white: the continuous version of "near-white".
    out["dist_to_white"] = np.sqrt(
        (out["lab_L"] - LAB_WHITE[0]) ** 2
        + (out["lab_a"] - LAB_WHITE[1]) ** 2
        + (out["lab_b"] - LAB_WHITE[2]) ** 2
    )
    # Chroma / hue: neutrals vs saturated colours behave very differently.
    out["lab_chroma"] = np.sqrt(out["lab_a"] ** 2 + out["lab_b"] ** 2)
    out["lab_hue"] = np.degrees(np.arctan2(out["lab_b"], out["lab_a"])) % 360
    out["is_neutral"] = out["lab_chroma"] < 8
    return out
