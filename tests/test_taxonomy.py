"""Colour taxonomy must resolve every catalogue colour, with no silent fallbacks."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.taxonomy import annotate, parse_colour


def test_every_catalogue_colour_resolves():
    """A colour that falls through to 'Other'/'unknown' silently becomes a
    neutral mid-grey, which would corrupt the Lab features it feeds."""
    p = Path("data/processed/sku_annotated.parquet")
    if not p.exists():
        pytest.skip("panel not built")
    sku = pd.read_parquet(p)
    assert (sku.shade_family == "Other").sum() == 0
    assert (sku.base_colour == "unknown").sum() == 0


@pytest.mark.parametrize("colour,family,pattern", [
    ("White", "White", "Solid"),
    ("Antique White", "Near White", "Solid"),
    ("Textured - White Waffle", "White", "Textured"),
    ("Pattern - Polka Dot", "White", "Pattern"),
    ("Stripe - Charcoal Gray", "Dark Grey", "Stripe"),
    ("White/Gray", "White", "Two Tone"),
    ("Indigo Navy Blue", "Dark Blue", "Solid"),
    ("Silver Sage", "Green", "Solid"),
])
def test_known_mappings(colour, family, pattern):
    a = parse_colour(colour)
    assert a.shade_family == family
    assert a.pattern_type == pattern


def test_longest_match_wins():
    """'indigo navy blue' must not be captured by the bare 'blue' rule."""
    assert parse_colour("Indigo Navy Blue").shade_family == "Dark Blue"
    assert parse_colour("Indigo Dusty Blue").shade_family == "Mid Blue"


def test_white_is_lab_origin():
    a = annotate(pd.DataFrame({"colour": ["White", "Black"]}))
    assert a.loc[0, "dist_to_white"] == pytest.approx(0.0, abs=1e-6)
    assert a.loc[1, "dist_to_white"] > 50
