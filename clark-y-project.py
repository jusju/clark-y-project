#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clark Y (≈11.7–13 %), flat-bottom from x/c >= x_flat, SVG exporter.

WHAT'S FIXED (compared to the user's previous version):
1) Leading edge shape: we keep the airfoil x-monotone and *do not*
   apply a closed Catmull–Rom spline around the outline (that caused
   overshoot/flattening near the nose). We emit a dense polyline instead,
   which preserves the analytic shape exactly.
2) Lower surface transition at x_flat uses a C1-continuous blend
   (Smoothstep h(s)=s²(3-2s)), so y_lower and its slope go to zero at
   x_flat. No kink there; the front looks like a true Clark Y.
3) Thickness ratio is the usual Clark Y ballpark (default t=0.117).
   You can change to 0.130 if you really want a “thicker Clark-Y-like”.
4) Clean CSV + SVG with chord in millimetres.

Outputs:
- SVG: clarky_200mm.svg
- CSV: clarky_200mm_coords.csv
and console prints camber max (%, location).

Usage: just run the file.
"""

from __future__ import annotations
import math
import csv
from typing import Tuple, List
import numpy as np

# -------------------------------
# Parameters you may want to edit
# -------------------------------
CHORD_MM   = 200.0    # chord length in mm
T_REL      = 0.117    # thickness ratio (Clark Y ≈ 11.7%)
X_FLAT     = 0.30     # flat lower surface starts here (x/c >= X_FLAT)
NPTS_X     = 800      # number of x-samples (half-curve -> dense polyline)
STROKE_W   = 0.30     # svg stroke width (mm)

# -------------------------------------------
# Thickness function: NACA 00-series half-th.
# y_t(x; t) returns *half*-thickness (in chord units)
# -------------------------------------------
def naca00_y_t(x: np.ndarray, t: float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (t / 0.2) * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
        + 0.2843 * x**3 - 0.1015 * x**4
    )

def smoothstep(s: np.ndarray) -> np.ndarray:
    """h(s)=s^2(3-2s),  h(0)=0,h'(0)=0, h(1)=1,h'(1)=0."""
    return np.clip(s, 0.0, 1.0)**2 * (3.0 - 2.0 * np.clip(s, 0.0, 1.0))

def clark_y_like(chord_mm: float = CHORD_MM,
                 t_rel: float = T_REL,
                 x_flat: float = X_FLAT,
                 npts: int = NPTS_X) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a Clark Y–like profile:
      - thickness distribution: NACA 00-xx (good LE radius)
      - lower surface is flat from x >= x_flat
      - in the nose (x < x_flat) we blend the camberline so that
        y_lower goes smoothly (C1) to zero at x_flat.
    Returns x_mm, y_upper_mm, y_lower_mm, camber_mm (all numpy arrays).
    """
    # cosine spacing: denser near LE and TE
    i = np.arange(npts + 1)
    x = 0.5 * (1.0 - np.cos(np.pi * i / npts))  # 0..1

    yt = naca00_y_t(x, t_rel)                    # half thickness (chord units)

    # Camber: choose it so that y_lower = 0 for x >= x_flat;
    # blend smoothly to the nose using smoothstep.
    camber = np.empty_like(x)
    aft = x >= x_flat
    fore = ~aft
    camber[aft] = yt[aft]
    s = x[fore] / x_flat
    h = smoothstep(s)
    camber[fore] = yt[fore] * h  # ensures y_lower' = 0 at x_flat

    y_upper = camber + yt
    y_lower = camber - yt

    # scale to mm
    return x * chord_mm, y_upper * chord_mm, y_lower * chord_mm, camber * chord_mm

def make_svg_poly_path(points: List[Tuple[float, float]], close: bool = True) -> str:
    """SVG 'd' path using polyline segments (no overshooting splines)."""
    if not points:
        return ""
    d = [f"M {points[0][0]:.4f},{points[0][1]:.4f}"]
    for x, y in points[1:]:
        d.append(f"L {x:.4f},{y:.4f}")
    if close:
        d.append("Z")
    return " ".join(d)

def build_outline(x_mm: np.ndarray, y_u_mm: np.ndarray, y_l_mm: np.ndarray) -> List[Tuple[float, float]]:
    """
    Closed outline TE -> LE (upper) and LE -> TE (lower).
    """
    upper = list(zip(x_mm[::-1], y_u_mm[::-1]))          # TE -> LE
    lower = list(zip(x_mm[1:], y_l_mm[1:]))              # LE -> TE (skip duplicated LE point)
    return upper + lower

def export_svg(filename: str,
               outline: List[Tuple[float, float]],
               chord_mm: float,
               stroke_w: float = STROKE_W,
               extra_paths: List[str] | None = None,
               padding: float = 0.05) -> None:
    xs = np.array([p[0] for p in outline])
    ys = np.array([p[1] for p in outline])
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    width, height = maxx - minx, maxy - miny
    margin = max(width, height) * padding
    viewbox = (minx - margin, miny - margin, width + 2*margin, height + 2*margin)

    d_outline = make_svg_poly_path(outline, close=True)

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}mm" height="{height:.3f}mm"',
        f'     viewBox="{viewbox[0]:.3f} {viewbox[1]:.3f} {viewbox[2]:.3f} {viewbox[3]:.3f}" version="1.1">',
        '  <g fill="none" stroke="black" stroke-linejoin="round" stroke-width="{:.3f}">'.format(stroke_w),
        f'    <path d="{d_outline}" />',
        f'    <line x1="0" y1="0" x2="{chord_mm:.3f}" y2="0" stroke="gray" stroke-width="{stroke_w/2:.3f}" stroke-dasharray="1,1"/>',
    ]
    if extra_paths:
        svg_parts.extend(extra_paths)
    svg_parts.append('  </g>')
    svg_parts.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))

def export_csv(filename: str,
               x_mm: np.ndarray, y_u_mm: np.ndarray, y_l_mm: np.ndarray, camber_mm: np.ndarray) -> None:
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x_mm", "y_upper_mm", "y_lower_mm", "camber_mm"])
        for i in range(len(x_mm)):
            w.writerow([f"{x_mm[i]:.6f}", f"{y_u_mm[i]:.6f}", f"{y_l_mm[i]:.6f}", f"{camber_mm[i]:.6f}"])

def main():
    x_mm, y_u_mm, y_l_mm, camber_mm = clark_y_like(CHORD_MM, T_REL, X_FLAT, NPTS_X)

    # Camber info
    camber_pct = float(camber_mm.max() / CHORD_MM * 100.0)
    x_at_max   = float(x_mm[camber_mm.argmax()] / CHORD_MM)

    # Build closed outline and export SVG/CSV
    outline = build_outline(x_mm, y_u_mm, y_l_mm)
    export_svg("clarky_200mm.svg", outline, CHORD_MM)
    export_csv("clarky_200mm_coords.csv", x_mm, y_u_mm, y_l_mm, camber_mm)

    print("SVG: clarky_200mm.svg")
    print("CSV: clarky_200mm_coords.csv")
    print(f"CAMBER ≈ {camber_pct:.2f}% @ x/c ≈ {x_at_max:.3f}")
    print(f"THICKNESS (t/c): {T_REL:.3f}, flat bottom from x/c ≥ {X_FLAT:.2f}")
    # A hint for the user:
    print("Tip: set T_REL=0.130 if you want a thicker Clark Y–like profile.")

if __name__ == "__main__":
    main()
