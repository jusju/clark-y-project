#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate-hansa-ribs.py
======================
Tuottaa yhden SVG-tiedoston, joka sisältää laserleikkausta varten:
  1) 20 siipikaarta (modifioitu Clark Y, t=10%, m=4.5%) — TE-linja suora,
     LE siirtyy taaksepäin 1 mm/kaari, jänne 200→181 mm.
     Kullakin kaarella johtoreunan katkaisu 4 mm, jättöreunan katkaisu 30 mm,
     yläpinnan torsioura 100 x 1 mm sekä kaksi Ø7 mm reikää (110 ja 150 mm
     alkup. jättöreunasta).
  2) Kaksi rimavanteen aihiota oikealle puolelle:
        LE 4 x 1000 mm,  TE 30 x 1000 mm,  10 mm tyhjää välissä.
  3) 80 suorakulmaista kolmiota (kateetit 20 mm) kahdessa pystyrivissä,
     40 / rivi, 5 mm väli kolmioiden välissä.
  4) Jokaisen kappaleen leikkausviivassa on 2 mm:n pituisia "siltoja"
     (mikrokiinnitykset), jotka pitävät kappaleet kiinni levyssä
     leikkauksen aikana.

Käyttö:
    python3 generate-hansa-ribs.py
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

# ----------------------------------------------------------------------
# PROFIILIPARAMETRIT
# ----------------------------------------------------------------------
FLAT_BOTTOM = False
T_REL  = 0.10
M_CAM  = 0.045
P_CAM  = 0.30
X_FLAT = 0.30
NPTS   = 400

# ----------------------------------------------------------------------
# SIIPIKAARIPARAMETRIT
# ----------------------------------------------------------------------
N_RIBS         = 20
CHORD_ROOT_MM  = 200.0
CHORD_STEP_MM  = 1.0

LE_RIM_WIDTH_MM = 4.0
TE_RIM_WIDTH_MM = 30.0

SLOT_WIDTH_MM = 100.0
SLOT_DEPTH_MM = 1.0

HOLE_DIAM_MM             = 7.0
HOLE_POSITIONS_FROM_TE_MM = [110.0, 150.0]   # 11 cm ja 15 cm alkup. jättöreunasta

# ----------------------------------------------------------------------
# RIMOJEN AIHIOT
# ----------------------------------------------------------------------
RIM_STOCK_LENGTH_MM       = 1000.0
RIM_GAP_FROM_RIBS_MM      = 10.0
RIM_GAP_BETWEEN_STOCKS_MM = 10.0

# ----------------------------------------------------------------------
# SUORAKULMAISET KOLMIOT
# ----------------------------------------------------------------------
TRIANGLE_LEG_MM           = 20.0    # molemmat kateetit
N_TRIANGLES               = 80
TRIANGLE_GAP_MM           = 5.0     # väli kolmioiden välissä (pysty)
TRIANGLE_COL_GAP_MM       = 5.0     # väli kolmiopystyrivien välissä
TRIANGLE_GAP_FROM_RIMS_MM = 10.0    # väli rimoista kolmioihin
TRIANGLE_COLUMN_LENGTH_MM = 1000.0  # tavoitteena 1 m pystyriviä kohden

# ----------------------------------------------------------------------
# LASER-MIKROKIINNITYKSET (siltojen pituus = leikkausviivan aukon koko)
# ----------------------------------------------------------------------
TAB_LENGTH_MM = 2.0       # 2 mm aukko leikkausviivassa per silta

# Per kappale: kuinka monta siltaa pitkien sivujen joukossa / kpl
RIB_TAB_FRACTIONS         = [0.04, 0.60, 0.75, 0.92]  # 1 yläpinta + 3 alapinta
RIM_LE_TABS_PER_LONG_SIDE = 4   # ohut & pitkä — riittävä määrä siltoja tärkeä
RIM_TE_TABS_PER_LONG_SIDE = 4
TRIANGLE_TABS_PER_LEG     = 1   # 2 siltaa per kolmio (yksi kummallakin kateetilla)

# ----------------------------------------------------------------------
# SVG-ASETTELU
# ----------------------------------------------------------------------
PAGE_MARGIN_MM      = 10.0
RIB_PITCH_MM        = 50.0
STROKE_W_MM         = 0.20
LABEL_FONT_SIZE_MM  = 8.0


# ======================================================================
# PROFIILI
# ======================================================================
def naca_thickness(x: np.ndarray, t: float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (t / 0.2) * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                        + 0.2843*x**3 - 0.1015*x**4)


def naca_camber(x: np.ndarray, m: float, p: float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    yc = np.empty_like(x)
    fwd = x < p
    aft = ~fwd
    yc[fwd] = (m / p**2)       * (2*p*x[fwd] - x[fwd]**2)
    yc[aft] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*x[aft] - x[aft]**2)
    return yc


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0); return s*s*(3.0 - 2.0*s)


def airfoil_xy(chord_mm, npts=NPTS):
    i = np.arange(npts + 1)
    x_rel = 0.5 * (1.0 - np.cos(np.pi * i / npts))
    yt = naca_thickness(x_rel, T_REL)
    if FLAT_BOTTOM:
        yc = np.empty_like(x_rel)
        aft = x_rel >= X_FLAT; fwd = ~aft
        yc[aft] = yt[aft]
        yc[fwd] = yt[fwd] * smoothstep(x_rel[fwd] / X_FLAT)
    else:
        yc = naca_camber(x_rel, M_CAM, P_CAM)
    return (x_rel*chord_mm, (yc+yt)*chord_mm, (yc-yt)*chord_mm, yc*chord_mm)


# ======================================================================
# KAAREN ÄÄRIVIIVA
# ======================================================================
def rib_outline(chord_mm: float) -> List[Tuple[float, float]]:
    x_mm, y_up, y_lo, _ = airfoil_xy(chord_mm)
    x_LE    = LE_RIM_WIDTH_MM
    x_TE    = chord_mm - TE_RIM_WIDTH_MM
    x_slotA = LE_RIM_WIDTH_MM
    x_slotB = LE_RIM_WIDTH_MM + SLOT_WIDTH_MM

    y_up_LE    = float(np.interp(x_LE,    x_mm, y_up))
    y_lo_LE    = float(np.interp(x_LE,    x_mm, y_lo))
    y_up_TE    = float(np.interp(x_TE,    x_mm, y_up))
    y_lo_TE    = float(np.interp(x_TE,    x_mm, y_lo))
    y_up_slotB = float(np.interp(x_slotB, x_mm, y_up))

    pts: List[Tuple[float, float]] = []
    pts.append((x_TE, y_up_TE))

    mask = (x_mm > x_slotB) & (x_mm < x_TE)
    for x, y in sorted(zip(x_mm[mask], y_up[mask]), key=lambda p: -p[0]):
        pts.append((float(x), float(y)))

    pts.append((x_slotB, y_up_slotB))
    pts.append((x_slotB, y_up_slotB - SLOT_DEPTH_MM))

    mask = (x_mm > x_slotA) & (x_mm < x_slotB)
    for x, y in sorted(zip(x_mm[mask], y_up[mask] - SLOT_DEPTH_MM),
                       key=lambda p: -p[0]):
        pts.append((float(x), float(y)))

    pts.append((x_LE, y_up_LE - SLOT_DEPTH_MM))
    pts.append((x_LE, y_lo_LE))

    mask = (x_mm > x_LE) & (x_mm < x_TE)
    for x, y in sorted(zip(x_mm[mask], y_lo[mask]), key=lambda p: p[0]):
        pts.append((float(x), float(y)))

    pts.append((x_TE, y_lo_TE))
    return pts


# ======================================================================
# LASER-MIKROKIINNITYKSET
# ======================================================================
def emit_paths_with_tabs(pts: List[Tuple[float, float]],
                         tab_arc_positions: List[float],
                         tab_length: float = TAB_LENGTH_MM) -> List[str]:
    """
    Suljetusta monikulmiosta palauttaa listan SVG-polkujen 'd'-merkkijonoja
    siten, että jokaisessa tab_arc_positions[i]-arvossa leikkausviivassa on
    tab_length-mittainen aukko (mikrokiinnitys).
    """
    if not pts or len(pts) < 3:
        return []

    pts_arr = np.array(pts, dtype=float)
    n = len(pts_arr)
    edges_end = np.roll(pts_arr, -1, axis=0)
    seg_lens = np.linalg.norm(edges_end - pts_arr, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total <= 0:
        return []

    if not tab_arc_positions:
        d = f"M {pts[0][0]:.4f},{pts[0][1]:.4f}"
        for x, y in pts[1:]:
            d += f" L {x:.4f},{y:.4f}"
        d += " Z"
        return [d]

    def point_at(arc):
        arc = arc % total
        i = int(np.searchsorted(cum, arc, side='right') - 1)
        i = max(0, min(i, n-1))
        if seg_lens[i] == 0:
            return (float(pts_arr[i][0]), float(pts_arr[i][1]))
        t = (arc - cum[i]) / seg_lens[i]
        p = pts_arr[i] + t * (edges_end[i] - pts_arr[i])
        return (float(p[0]), float(p[1]))

    tabs = sorted([
        ((ta - tab_length/2.0) % total, (ta + tab_length/2.0) % total)
        for ta in tab_arc_positions
    ], key=lambda x: x[0])

    k = len(tabs)
    sub_paths_d: List[str] = []
    for i in range(k):
        seg_start = tabs[i][1]                    # aukon loppu
        seg_end   = tabs[(i+1) % k][0]            # seuraavan aukon alku
        sub: List[Tuple[float, float]] = [point_at(seg_start)]
        if seg_start < seg_end:
            for j in range(n):
                a = cum[j]
                if seg_start < a < seg_end:
                    sub.append((float(pts_arr[j][0]), float(pts_arr[j][1])))
        else:
            for j in range(n):
                a = cum[j]
                if a > seg_start:
                    sub.append((float(pts_arr[j][0]), float(pts_arr[j][1])))
            for j in range(n):
                a = cum[j]
                if a < seg_end:
                    sub.append((float(pts_arr[j][0]), float(pts_arr[j][1])))
        sub.append(point_at(seg_end))
        if len(sub) >= 2:
            d = f"M {sub[0][0]:.4f},{sub[0][1]:.4f}"
            for x, y in sub[1:]:
                d += f" L {x:.4f},{y:.4f}"
            sub_paths_d.append(d)
    return sub_paths_d


def _perimeter(pts):
    pts_arr = np.array(pts, dtype=float)
    edges_end = np.roll(pts_arr, -1, axis=0)
    return float(np.linalg.norm(edges_end - pts_arr, axis=1).sum())


def tabs_rib(pts) -> List[float]:
    P = _perimeter(pts)
    return [f * P for f in RIB_TAB_FRACTIONS]


def tabs_triangle(leg: float) -> List[float]:
    """Tabs: midpoint of bottom leg and midpoint of left leg
    (so right-angle vertex stays held by surrounding material; hypotenuse fully cut).
    Polygon: (0,0)->(leg,0)->(0,leg) closed.
    Edges in order: bottom(L), hyp(L*sqrt2), left(L)."""
    hyp = leg * np.sqrt(2.0)
    return [
        leg * 0.5,                       # bottom-leg midpoint (arc=leg/2)
        leg + hyp + leg * 0.5,           # left-leg midpoint
    ]


def tabs_rectangle(w: float, h: float, n_per_long: int = 4) -> List[float]:
    """For rectangle (0,0)->(w,0)->(w,h)->(0,h), n_per_long tabs per long side
    (the two h-long vertical sides)."""
    tabs = []
    for i in range(n_per_long):
        frac = (i + 0.5) / n_per_long
        tabs.append(w + frac * h)              # right vertical edge
        tabs.append(2.0 * w + h + frac * h)    # left vertical edge
    return tabs


# ======================================================================
# SVG
# ======================================================================
def _triangle_layout():
    """Returns (per_col, n_cols) for current N_TRIANGLES & column length."""
    pitch = TRIANGLE_LEG_MM + TRIANGLE_GAP_MM
    per_col = int((TRIANGLE_COLUMN_LENGTH_MM + TRIANGLE_GAP_MM) // pitch)
    per_col = max(1, per_col)
    n_cols = -(-N_TRIANGLES // per_col)        # ceil
    return per_col, n_cols


def total_width() -> float:
    rims = (RIM_GAP_FROM_RIBS_MM + LE_RIM_WIDTH_MM
            + RIM_GAP_BETWEEN_STOCKS_MM + TE_RIM_WIDTH_MM)
    per_col, n_cols = _triangle_layout()
    tri_w = (n_cols * TRIANGLE_LEG_MM
             + max(0, n_cols - 1) * TRIANGLE_COL_GAP_MM)
    return PAGE_MARGIN_MM * 2 + CHORD_ROOT_MM + rims + TRIANGLE_GAP_FROM_RIMS_MM + tri_w


def total_height() -> float:
    ribs_h = RIB_PITCH_MM * N_RIBS
    rims_h = RIM_STOCK_LENGTH_MM
    return PAGE_MARGIN_MM * 2 + max(ribs_h, rims_h, TRIANGLE_COLUMN_LENGTH_MM)


def build_svg() -> str:
    te_x = PAGE_MARGIN_MM + CHORD_ROOT_MM
    body: List[str] = []

    # --- 20 kaarta ---
    body.append('    <!-- Siipikaaret -->')
    for k in range(1, N_RIBS + 1):
        chord = CHORD_ROOT_MM - (k - 1) * CHORD_STEP_MM
        outline = rib_outline(chord)
        x_shift = te_x - chord
        y_center = PAGE_MARGIN_MM + RIB_PITCH_MM * (k - 0.5)
        svg_pts = [(p[0] + x_shift, y_center - p[1]) for p in outline]

        for d in emit_paths_with_tabs(svg_pts, tabs_rib(svg_pts)):
            body.append(f'    <path d="{d}" />')

        # reiät (ei siltoja näihin: jätepalat ovat niin pieniä että putoavat
        # joka tapauksessa hallitusti)
        x_mm_c, _, _, yc = airfoil_xy(chord)
        for d_te in HOLE_POSITIONS_FROM_TE_MM:
            x_hole = chord - d_te
            yc_h = float(np.interp(x_hole, x_mm_c, yc))
            cx = x_hole + x_shift
            cy = y_center - yc_h
            body.append(f'    <circle cx="{cx:.4f}" cy="{cy:.4f}" '
                        f'r="{HOLE_DIAM_MM/2:.4f}" />')

        # numero
        x_label = 0.30 * chord
        yc_l = float(np.interp(x_label, x_mm_c, yc))
        lx = x_label + x_shift
        ly = y_center - yc_l + LABEL_FONT_SIZE_MM * 0.35
        body.append(
            f'    <text x="{lx:.3f}" y="{ly:.3f}" '
            f'font-family="sans-serif" font-size="{LABEL_FONT_SIZE_MM:.2f}" '
            f'text-anchor="middle" fill="black" stroke="none">{k}</text>'
        )

    # --- Rimojen aihiot ---
    body.append('    <!-- Rimojen aihiot -->')
    rim_y_top = PAGE_MARGIN_MM
    rim_le_x  = te_x + RIM_GAP_FROM_RIBS_MM
    rim_te_x  = rim_le_x + LE_RIM_WIDTH_MM + RIM_GAP_BETWEEN_STOCKS_MM

    le_rect = [
        (rim_le_x,                    rim_y_top),
        (rim_le_x + LE_RIM_WIDTH_MM,  rim_y_top),
        (rim_le_x + LE_RIM_WIDTH_MM,  rim_y_top + RIM_STOCK_LENGTH_MM),
        (rim_le_x,                    rim_y_top + RIM_STOCK_LENGTH_MM),
    ]
    for d in emit_paths_with_tabs(le_rect,
                                  tabs_rectangle(LE_RIM_WIDTH_MM,
                                                 RIM_STOCK_LENGTH_MM,
                                                 RIM_LE_TABS_PER_LONG_SIDE)):
        body.append(f'    <path d="{d}" />')

    te_rect = [
        (rim_te_x,                    rim_y_top),
        (rim_te_x + TE_RIM_WIDTH_MM,  rim_y_top),
        (rim_te_x + TE_RIM_WIDTH_MM,  rim_y_top + RIM_STOCK_LENGTH_MM),
        (rim_te_x,                    rim_y_top + RIM_STOCK_LENGTH_MM),
    ]
    for d in emit_paths_with_tabs(te_rect,
                                  tabs_rectangle(TE_RIM_WIDTH_MM,
                                                 RIM_STOCK_LENGTH_MM,
                                                 RIM_TE_TABS_PER_LONG_SIDE)):
        body.append(f'    <path d="{d}" />')

    # Pienet otsikkotekstit rimojen yläpäähän
    label_y = rim_y_top - 1.5
    body.append(f'    <text x="{rim_le_x + LE_RIM_WIDTH_MM/2:.3f}" '
                f'y="{label_y:.3f}" font-family="sans-serif" font-size="3.0" '
                f'text-anchor="middle" fill="black" stroke="none">LE 4x1000</text>')
    body.append(f'    <text x="{rim_te_x + TE_RIM_WIDTH_MM/2:.3f}" '
                f'y="{label_y:.3f}" font-family="sans-serif" font-size="3.0" '
                f'text-anchor="middle" fill="black" stroke="none">TE 30x1000</text>')

    # --- 80 kolmiota ---
    body.append('    <!-- Suorakulmaiset kolmiot -->')
    per_col, n_cols = _triangle_layout()
    tri_first_col_x = (te_x + RIM_GAP_FROM_RIBS_MM + LE_RIM_WIDTH_MM
                       + RIM_GAP_BETWEEN_STOCKS_MM + TE_RIM_WIDTH_MM
                       + TRIANGLE_GAP_FROM_RIMS_MM)

    for ti in range(N_TRIANGLES):
        col = ti // per_col
        row = ti % per_col
        col_x = tri_first_col_x + col * (TRIANGLE_LEG_MM + TRIANGLE_COL_GAP_MM)
        row_y = PAGE_MARGIN_MM + row * (TRIANGLE_LEG_MM + TRIANGLE_GAP_MM)
        tri_pts = [
            (col_x,                       row_y),
            (col_x + TRIANGLE_LEG_MM,     row_y),
            (col_x,                       row_y + TRIANGLE_LEG_MM),
        ]
        for d in emit_paths_with_tabs(tri_pts, tabs_triangle(TRIANGLE_LEG_MM)):
            body.append(f'    <path d="{d}" />')

    # --- SVG ---
    W = total_width(); H = total_height()
    svg = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        '<svg xmlns="http://www.w3.org/2000/svg"',
        f'     width="{W:.3f}mm" height="{H:.3f}mm"',
        f'     viewBox="0 0 {W:.3f} {H:.3f}" version="1.1">',
        f'  <g fill="none" stroke="black" stroke-width="{STROKE_W_MM:.3f}" '
        'stroke-linejoin="round" stroke-linecap="round">',
        *body,
        '  </g>',
        '</svg>',
    ]
    return "\n".join(svg)


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    svg_text = build_svg()
    out_path = "hansa-ribs.svg"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg_text)

    x_mm, y_up, y_lo, yc = airfoil_xy(CHORD_ROOT_MM)
    max_camber_pct = float(yc.max() / CHORD_ROOT_MM * 100.0)
    max_thick_pct  = float((y_up - y_lo).max() / CHORD_ROOT_MM * 100.0)
    per_col, n_cols = _triangle_layout()
    chord_tip = CHORD_ROOT_MM - (N_RIBS - 1) * CHORD_STEP_MM

    print(f"Tallennettu {out_path}")
    print(f"  Profiili: t/c={T_REL:.3f}  m/c={max_camber_pct/100:.3f}")
    print(f"  Mitattu:  paksuus={max_thick_pct:.2f}%  kamberi={max_camber_pct:.2f}%")
    print(f"  Kaaria:   {N_RIBS} kpl, jänne {CHORD_ROOT_MM:.0f}→{chord_tip:.0f} mm")
    print(f"  Rimat:    LE {LE_RIM_WIDTH_MM:.0f}x{RIM_STOCK_LENGTH_MM:.0f}, "
          f"TE {TE_RIM_WIDTH_MM:.0f}x{RIM_STOCK_LENGTH_MM:.0f} mm")
    print(f"  Kolmiot:  {N_TRIANGLES} kpl, {per_col} kpl/pystyrivi x "
          f"{n_cols} riviä, kateetit {TRIANGLE_LEG_MM:.0f} mm")
    print(f"  Sillat:   {TAB_LENGTH_MM:.1f} mm aukot leikkausviivassa")
    print(f"    - per kaari: {len(RIB_TAB_FRACTIONS)} kpl")
    print(f"    - per LE-rima: {2*RIM_LE_TABS_PER_LONG_SIDE} kpl "
          f"(pitkillä sivuilla)")
    print(f"    - per TE-rima: {2*RIM_TE_TABS_PER_LONG_SIDE} kpl "
          f"(pitkillä sivuilla)")
    print(f"    - per kolmio: 2 kpl (toinen molemmalla kateetilla)")
    print(f"  Sivu:     {total_width():.0f} x {total_height():.0f} mm")


if __name__ == "__main__":
    main()
