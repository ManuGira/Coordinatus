"""Unit tests for coordinatus.viz.color_hash."""

from __future__ import annotations

import pytest

from coordinatus.viz.color_hash import generate, hsl2rgb, rgb2hsl, to_hex


# --------------------------------------------------------------------------- #
# to_hex
# --------------------------------------------------------------------------- #

class TestToHex:
    def test_black(self):
        assert to_hex(0, 0, 0) == "#000000"

    def test_white(self):
        assert to_hex(255, 255, 255) == "#ffffff"

    def test_red(self):
        assert to_hex(255, 0, 0) == "#ff0000"

    def test_green(self):
        assert to_hex(0, 255, 0) == "#00ff00"

    def test_blue(self):
        assert to_hex(0, 0, 255) == "#0000ff"

    def test_mixed(self):
        assert to_hex(16, 32, 48) == "#102030"

    def test_single_digit_hex_padded(self):
        assert to_hex(1, 2, 3) == "#010203"


# --------------------------------------------------------------------------- #
# rgb2hsl
# --------------------------------------------------------------------------- #

class TestRgb2Hsl:
    def test_black(self):
        hue, s, lum = rgb2hsl(0, 0, 0)
        assert hue == 0
        assert s == 0
        assert lum == 0

    def test_white(self):
        hue, s, lum = rgb2hsl(1, 1, 1)
        assert s == 0
        assert lum == 100

    def test_achromatic_gives_zero_saturation(self):
        hue, s, lum = rgb2hsl(0.5, 0.5, 0.5)
        assert s == 0

    def test_red(self):
        hue, s, lum = rgb2hsl(1, 0, 0)
        assert hue == 0
        assert s == 100

    def test_green(self):
        hue, s, lum = rgb2hsl(0, 1, 0)
        assert hue == 120
        assert s == 100

    def test_blue(self):
        hue, s, lum = rgb2hsl(0, 0, 1)
        assert hue == 240
        assert s == 100

    def test_hue_wraps_for_blue_dominant(self):
        # When r is max but g < b the hue offset adds 360 degrees worth of shift
        hue, s, lum = rgb2hsl(0.9, 0.1, 0.5)
        assert 0 <= hue <= 360

    def test_returns_ints(self):
        hue, s, lum = rgb2hsl(0.3, 0.6, 0.9)
        assert isinstance(hue, int)
        assert isinstance(s, int)
        assert isinstance(lum, int)


# --------------------------------------------------------------------------- #
# hsl2rgb
# --------------------------------------------------------------------------- #

class TestHsl2Rgb:
    def test_black(self):
        assert hsl2rgb(0, 0, 0) == (0, 0, 0)

    def test_white(self):
        assert hsl2rgb(0, 0, 100) == (255, 255, 255)

    def test_achromatic_saturation_zero(self):
        r, g, b = hsl2rgb(180, 0, 50)
        assert r == g == b

    def test_red(self):
        r, g, b = hsl2rgb(0, 100, 50)
        assert r == 255
        assert g == 0
        assert b == 0

    def test_green(self):
        r, g, b = hsl2rgb(120, 100, 50)
        assert r == 0
        assert g == 255
        assert b == 0

    def test_blue(self):
        r, g, b = hsl2rgb(240, 100, 50)
        assert r == 0
        assert g == 0
        assert b == 255

    def test_returns_ints(self):
        r, g, b = hsl2rgb(200, 60, 50)
        assert isinstance(r, int)
        assert isinstance(g, int)
        assert isinstance(b, int)

    def test_values_in_byte_range(self):
        for hue in range(0, 361, 60):
            r, g, b = hsl2rgb(hue, 80, 50)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255


# --------------------------------------------------------------------------- #
# rgb2hsl / hsl2rgb roundtrip
# --------------------------------------------------------------------------- #

class TestRoundtrip:
    """Converting RGB→HSL→RGB should recover the original colour (within
    integer rounding tolerance)."""

    @pytest.mark.parametrize("r,g,b", [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.5),
        (0.2, 0.4, 0.8),
        (0.9, 0.1, 0.5),
    ])
    def test_rgb_hsl_rgb(self, r, g, b):
        hue, s, lum = rgb2hsl(r, g, b)
        r2, g2, b2 = hsl2rgb(hue, s, lum)
        # integer rounding means we allow ±2 per channel
        assert abs(r2 - round(r * 255)) <= 2
        assert abs(g2 - round(g * 255)) <= 2
        assert abs(b2 - round(b * 255)) <= 2


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #

class TestGenerate:
    def test_returns_hex_string(self):
        result = generate("hello")
        assert isinstance(result, str)
        assert result.startswith("#")
        assert len(result) == 7

    def test_deterministic(self):
        assert generate("foo") == generate("foo")

    def test_different_seeds_produce_different_colours(self):
        colours = {generate(str(i)) for i in range(20)}
        # Very unlikely to have fewer than 15 distinct colours for 20 seeds
        assert len(colours) >= 15

    def test_saturation_in_range(self):
        """Generated colours should have saturation between 50% and 100%."""
        for seed in ["a", "b", "test", "coordinatus", "123"]:
            hex_colour = generate(seed)
            r = int(hex_colour[1:3], 16)
            g = int(hex_colour[3:5], 16)
            b = int(hex_colour[5:7], 16)
            _, s, _ = rgb2hsl(r / 255, g / 255, b / 255)
            assert 45 <= s <= 100, f"saturation {s} out of range for seed {seed!r}"

    def test_lightness_in_range(self):
        """Generated colours should have lightness between 40% and 60%."""
        for seed in ["a", "b", "test", "coordinatus", "123"]:
            hex_colour = generate(seed)
            r = int(hex_colour[1:3], 16)
            g = int(hex_colour[3:5], 16)
            b = int(hex_colour[5:7], 16)
            _, _, lum = rgb2hsl(r / 255, g / 255, b / 255)
            assert 35 <= lum <= 65, f"lightness {lum} out of range for seed {seed!r}"

    def test_empty_string_seed(self):
        result = generate("")
        assert result.startswith("#")
        assert len(result) == 7

    def test_unicode_seed(self):
        result = generate("héllo wörld")
        assert result.startswith("#")
        assert len(result) == 7
