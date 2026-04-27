"""3D visualization: why rebars appear as hyperbolas in GPR B-scans.

Left panel: 3D scene of the GPR antenna sweeping across a concrete slab
containing rebars, with rays from four antenna positions showing how
the round-trip travel time depends on horizontal offset.

Right panel: the resulting 2D B-scan (horizontal position vs. two-way
travel time) with hyperbolic echo traces.

Output: vector PDF at img/gpr/hyperbola_viz.pdf
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


OUT = Path(__file__).resolve().parent.parent / "img" / "gpr" / "hyperbola_viz.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Scene parameters ─────────────────────────────────────────────────────────
SLAB_LEN  = 10.0   # x
SLAB_WIDE = 4.0    # y
SLAB_DEEP = 5.0    # z (depth downward)

REBAR_Y = 2.0
REBAR_DEPTH = 2.5
REBAR_X  = [2.5, 5.0, 7.5]
REBAR_R  = 0.15
REBAR_LEN_Y = SLAB_WIDE - 0.4

ANT_XS = np.array([2.0, 3.5, 5.0, 6.5, 8.0])
ANT_Y = REBAR_Y
ANT_Z = 0.0

NAVYDARK = "#0D1B2A"
NAVYMID  = "#1B3A5C"
GOLD     = "#C9A84C"
MUTED    = "#6B7A8D"


def draw_slab(ax):
    x0, x1 = 0, SLAB_LEN
    y0, y1 = 0, SLAB_WIDE
    z0, z1 = 0, SLAB_DEEP  # z grows downward in our convention; we invert axis
    verts = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # top
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # bottom
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front (y=0)
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back  (y=max)
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right
    ]
    poly = Poly3DCollection(verts, alpha=0.08, facecolor=MUTED,
                            edgecolor=NAVYDARK, linewidths=0.5)
    ax.add_collection3d(poly)


def draw_rebar(ax, xc):
    # Cylinder along y-axis at (xc, REBAR_Y-axis, REBAR_DEPTH), radius REBAR_R
    theta = np.linspace(0, 2 * np.pi, 24)
    y_vals = np.linspace(0.2, 0.2 + REBAR_LEN_Y, 2)
    T, Y = np.meshgrid(theta, y_vals)
    X = xc + REBAR_R * np.cos(T)
    Z = REBAR_DEPTH + REBAR_R * np.sin(T)
    ax.plot_surface(X, Y, Z, color=GOLD, linewidth=0, alpha=0.95, shade=True)

    # End caps
    caps = []
    for yv in (y_vals[0], y_vals[-1]):
        pts = [(xc + REBAR_R * np.cos(t), yv, REBAR_DEPTH + REBAR_R * np.sin(t))
               for t in np.linspace(0, 2 * np.pi, 24)]
        caps.append(pts)
    ax.add_collection3d(Poly3DCollection(caps, color=GOLD, alpha=0.95))


def draw_antenna(ax, x):
    # Small box on the surface
    w, d, h = 0.45, 0.55, 0.25
    x0, x1 = x - w/2, x + w/2
    y0, y1 = ANT_Y - d/2, ANT_Y + d/2
    z0, z1 = -h, 0
    verts = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]
    ax.add_collection3d(Poly3DCollection(verts, color=NAVYDARK, alpha=0.9,
                                         edgecolor=NAVYDARK))


def draw_rays(ax):
    # For each antenna position, draw a ray to the nearest rebar
    lines = []
    colors = []
    for xa in ANT_XS:
        # pick the rebar with min horizontal offset
        xr = min(REBAR_X, key=lambda x: abs(x - xa))
        # down-going ray
        lines.append([(xa, ANT_Y, 0), (xr, ANT_Y, REBAR_DEPTH)])
        # up-going (reflected) ray offset slightly to distinguish
        lines.append([(xr, ANT_Y, REBAR_DEPTH), (xa, ANT_Y, 0)])
        colors.append(NAVYMID)
        colors.append(NAVYMID)
    lc = Line3DCollection(lines, colors=colors, linewidths=0.9, alpha=0.55)
    ax.add_collection3d(lc)


def make_3d_axes(fig, rect):
    ax = fig.add_subplot(rect, projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-62)

    draw_slab(ax)
    for xr in REBAR_X:
        draw_rebar(ax, xr)
    for xa in ANT_XS:
        draw_antenna(ax, xa)
    draw_rays(ax)

    # Antenna-sweep arrow
    ax.quiver(ANT_XS[0] - 0.6, ANT_Y, 0, ANT_XS[-1] - ANT_XS[0] + 1.2, 0, 0,
              arrow_length_ratio=0.04, color=NAVYDARK, linewidth=1.3)
    ax.text(ANT_XS[-1] + 0.8, ANT_Y, 0.1, "antenna sweep",
            color=NAVYDARK, fontsize=8)
    ax.text(REBAR_X[1] + 0.3, 0.2, REBAR_DEPTH + 0.2, "rebars",
            color=GOLD, fontsize=8, fontweight="bold")

    ax.set_xlim(0, SLAB_LEN)
    ax.set_ylim(0, SLAB_WIDE)
    ax.set_zlim(SLAB_DEEP, -1.0)  # invert so z=0 is top

    ax.set_xlabel("x (m)", fontsize=8, color=NAVYDARK)
    ax.set_ylabel("y (m)", fontsize=8, color=NAVYDARK)
    ax.set_zlabel("depth (m)", fontsize=8, color=NAVYDARK)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.pane.set_edgecolor("#FFFFFF")
        axis.line.set_linewidth(0.4)
        axis.line.set_color(MUTED)
    ax.tick_params(labelsize=7, colors=MUTED, pad=-3)
    ax.grid(False)
    ax.set_title("GPR sweep over rebars", color=NAVYDARK,
                 fontsize=10, fontweight="bold", pad=6)
    return ax


# ── 2D B-scan panel ──────────────────────────────────────────────────────────
def make_2d_axes(fig, rect):
    ax = fig.add_subplot(rect)
    xs = np.linspace(0, SLAB_LEN, 400)
    # Speed of light / sqrt(epsilon_r ~ 8) → v ≈ 0.106 m/ns, so use normalized time
    v = 1.0  # unitless for illustration
    for xr in REBAR_X:
        t = 2.0 * np.sqrt(REBAR_DEPTH**2 + (xs - xr)**2) / v
        ax.plot(xs, t, color=NAVYMID, linewidth=1.6)
        ax.plot(xr, 2.0 * REBAR_DEPTH / v, "o", color=GOLD, markersize=5,
                markeredgecolor=NAVYDARK, markeredgewidth=0.6)

    ax.set_xlim(0, SLAB_LEN)
    ax.set_ylim(14.5, 3.5)  # inverted time axis, matches radargram convention

    ax.set_xlabel("antenna position x (m)", fontsize=8, color=NAVYDARK)
    ax.set_ylabel("two-way travel time (a.u.)", fontsize=8, color=NAVYDARK)
    ax.tick_params(labelsize=7, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.5)
    ax.set_title("what appears on the screen", color=NAVYDARK,
                 fontsize=10, fontweight="bold", pad=6)
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, color="#D0D5DD")

    for xr in REBAR_X:
        ax.annotate("hyperbola", xy=(xr, 2.0 * REBAR_DEPTH / v + 0.2),
                    xytext=(xr + 1.0, 2.0 * REBAR_DEPTH / v + 2.2),
                    fontsize=7, color=NAVYDARK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
        break  # label only one
    return ax


# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 4.8))
fig.patch.set_facecolor("white")
make_3d_axes(fig, 121)
make_2d_axes(fig, 122)
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.10, wspace=0.25)
fig.savefig(OUT, format="pdf", bbox_inches="tight", pad_inches=0.1)
print(f"wrote {OUT}")
