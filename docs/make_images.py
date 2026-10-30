"""Generate the logo and showcase figures used in the README and documentation.

Run from the repository root with ``python docs/make_images.py``. Outputs are written
to ``docs/_static/images``.
"""

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import orthax  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), "_static", "images")
os.makedirs(OUTDIR, exist_ok=True)

# Text is converted to paths so the SVGs render the same without the font installed.
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["font.family"] = "DejaVu Sans"

# Warm light tile with gold/teal/coral curves, chosen to contrast with the purple and
# charcoal of the documentation sidebar.
BG = "#fdf6e3"
BG_EDGE = "#e6d6ac"
TEXT = "#2b2f30"
SUBTEXT = "#0d7a6d"
LOGO_COLORS = ["#e8505b", "#e0a100", "#0f9d8a"]
PLOT_COLORS = ["#e6337a", "#f08c00", "#1c9cc0", "#7c4dff", "#2a0040"]


def _basis(rec, n, x):
    """Evaluate the first n basis functions of rec at x, shape (n, len(x))."""
    return jnp.stack([orthax.orthval(x, jnp.eye(n)[i], rec) for i in range(n)])


def _draw_mark(ax, x0, y0, size, tile=True, scale=1.0, cy=None):
    """Draw the icon (Chebyshev curves on a rounded tile) in data coordinates."""
    pad = 0.08 * size
    if tile:
        ax.add_patch(
            FancyBboxPatch(
                (x0 + pad, y0 + pad),
                size - 2 * pad,
                size - 2 * pad,
                boxstyle=f"round,pad=0,rounding_size={0.18 * size}",
                facecolor=BG,
                edgecolor=BG_EDGE,
                linewidth=size * 1.5,
            )
        )
    degrees = [2, 3, 4]
    x = jnp.linspace(-1, 1, 400)
    T = _basis(orthax.recurrence.ChebyshevT(), max(degrees) + 1, x)
    inner = 0.28 * size * scale
    cx = x0 + size / 2
    cy = y0 + size / 2 if cy is None else cy
    for j, i in enumerate(degrees):
        ax.plot(
            cx + inner * x,
            cy + inner * T[i],
            color=LOGO_COLORS[j],
            linewidth=size * scale * 2.5,
            solid_capstyle="round",
        )


def make_logo():
    """Square icon, square logo with wordmark, and wide banner."""
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    _draw_mark(ax, 0, 0, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(os.path.join(OUTDIR, "logo_icon.svg"), transparent=True)
    fig.savefig(os.path.join(OUTDIR, "logo_icon.png"), transparent=True, dpi=128)
    plt.close(fig)

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    _draw_mark(ax, 0, 0, 1, scale=0.75, cy=0.6)
    ax.text(0.5, 0.17, "orthax", color=TEXT, fontsize=22, fontweight="bold", ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(os.path.join(OUTDIR, "logo.svg"), transparent=True)
    plt.close(fig)

    w, h = 5.0, 1.4
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.02),
            w - 0.04,
            h - 0.04,
            boxstyle="round,pad=0,rounding_size=0.25",
            facecolor=BG,
            edgecolor=BG_EDGE,
            linewidth=2,
        )
    )
    _draw_mark(ax, 0.1, 0.1, h - 0.2, tile=False)
    ax.text(
        h + 0.05,
        h * 0.58,
        "orthax",
        color=TEXT,
        fontsize=44,
        fontweight="bold",
        va="center",
    )
    ax.text(
        h + 0.08,
        h * 0.22,
        "orthogonal polynomials in JAX",
        color=SUBTEXT,
        fontsize=13,
        va="center",
    )
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(os.path.join(OUTDIR, "logo_banner.svg"), transparent=True)
    plt.close(fig)


def make_families():
    """Grid of normalized basis functions for several polynomial families."""
    rr = orthax.recurrence
    panels = [
        ("Legendre", rr.Legendre(scale="normalized"), (-1, 1)),
        ("Chebyshev (2nd kind)", rr.ChebyshevU(scale="normalized"), (-1, 1)),
        (
            r"Jacobi  $\alpha=2,\ \beta=-\frac{1}{2}$",
            rr.Jacobi(2.0, -0.5, scale="normalized"),
            (-1, 1),
        ),
        ("Hermite functions", rr.Hermite(scale="normalized"), (-5, 5)),
        ("Laguerre functions", rr.Laguerre(scale="normalized"), (0, 15)),
        (
            r"Custom: $w(x) = e^{-x^2}$ on $[0, \infty)$",
            rr.generate_recurrence(
                lambda x: jnp.exp(-(x**2)), (0, jnp.inf), n=8, scale="normalized"
            ),
            (0, 4),
        ),
    ]
    n = 5
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), layout="constrained")
    for ax, (title, rec, (a, b)) in zip(axes.flat, panels):
        x = jnp.linspace(a, b, 600)
        # Multiplying orthonormal polynomials by sqrt(weight) puts families with
        # different weights and domains on a comparable scale.
        P = _basis(rec, n, x) * jnp.sqrt(rec.weight(x))
        for i in range(n):
            ax.plot(x, P[i], color=PLOT_COLORS[i], lw=2, label=f"$n={i}$")
        ax.axhline(0, color="0.6", lw=0.6, zorder=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(a, b)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes.flat[0].legend(fontsize=8, frameon=False, ncol=1, loc="lower right")
    fig.suptitle(
        r"Orthonormal basis functions $\sqrt{w(x)}\,p_n(x)$", fontsize=13
    )
    fig.savefig(os.path.join(OUTDIR, "families.png"), dpi=110, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_logo()
    make_families()
