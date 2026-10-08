"""
Colours shared by the terminal view and the matplotlib plots.

Values (Q, V, rewards) use one diverging scale: red for negative, a neutral
gray at zero ("nothing learned yet"), blue for positive. Each arm has its own
extent, so 0 is always gray even though rewards run from -10 to +20.
"""

# Diverging scale, from the most negative value through zero to the most positive.
NEGATIVE_ARM = ["#9e2b2b", "#e34948", "#f1b5b3"]
MIDPOINT = "#f0efec"
POSITIVE_ARM = ["#9ec5f4", "#3987e5", "#184f95"]
DIVERGING_STOPS = NEGATIVE_ARM + [MIDPOINT] + POSITIVE_ARM

# Episode outcomes, in fixed categorical order.
OUTCOME_COLORS = {
    "+20": "#2a78d6",
    "+3": "#eb6834",
    "-10": "#1baf7a",
    "timeout": "#eda100",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
START = "#383835"        # the start square: neutral, off the value scale
PATH = "#0b0b0b"


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#" + "".join(f"{round(c):02x}" for c in rgb)


def _interpolate(stops, t):
    """Colour at position t in [0, 1] along evenly spaced hex `stops`."""
    t = min(max(t, 0.0), 1.0)
    scaled = t * (len(stops) - 1)
    i = min(int(scaled), len(stops) - 2)
    frac = scaled - i
    a, b = hex_to_rgb(stops[i]), hex_to_rgb(stops[i + 1])
    return rgb_to_hex(tuple(x + (y - x) * frac for x, y in zip(a, b)))


def diverging_color(value, vmin, vmax):
    """Hex colour for `value` on the diverging scale; vmin < 0 < vmax."""
    if value >= 0:
        return _interpolate([MIDPOINT] + POSITIVE_ARM, value / vmax if vmax > 0 else 0.0)
    return _interpolate([MIDPOINT] + NEGATIVE_ARM[::-1], value / vmin if vmin < 0 else 0.0)


def readable_ink(background_hex):
    """Black or white text, whichever reads better on `background_hex`."""
    r, g, b = (c / 255 for c in hex_to_rgb(background_hex))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if luminance > 0.5 else "#ffffff"
