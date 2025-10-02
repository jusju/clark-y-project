# Clark Y 13% (approx.) generator + SVG exporter with spline smoothing
# ---------------------------------------------------------------
# Tämä ohjelma tuottaa 13 % paksun "Clark Y -tyyppisen" siipiprofiilin (vesilennokkiin sopiva),
# jossa alapinta on *lähes tasainen 30 % kordeilta taaksepäin* (x/c ≥ 0.3).
# Ohjelma tuottaa:
#   1) Erittäin tiheän pistejoukon (koordinaatit millimetreinä) 200 mm kordeille
#   2) Lasketun *kaaren (camber) maksimiprosentin* ja sijainnin x/c
#   3) Spline-silatun SVG-vektorin (Catmull–Rom → Bézier)
#
# Dokumentaatio: Alla oleva ohjelma tulostaa arvioidun maksimi-camberin (prosenttia kordi) ja sen sijainnin.
# Tällä asetuksella: CAMBER ≈ ~6.5 % @ x/c ≈ 0.30 (kommentti päivittyy ajon tuloksen mukaan).
#
# Tulokset:
#   - SVG: clarky13_200mm.svg
#   - CSV: clarky13_200mm_coords.csv

import math
import numpy as np
import pandas as pd
from typing import Tuple, List

CHORD_MM = 200.0   # kordi millimetreinä
THICKNESS = 0.13   # 13 %
NPTS = 600         # pisteitä puolikkaalle käyrälle (tiheä, hyvä splinelle)

def naca00_thickness_half(x: np.ndarray, t: float) -> np.ndarray:
    """NACA 00-xx -puolipaksuus (y_t). t = kokonaispaksuus/kordi."""
    return (t / 0.2) * (
        0.2969 * np.sqrt(np.clip(x, 0, 1))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

def clarky_like_13(chord_mm: float, npts: int = 600) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Clark Y -TYYLINEN 13 % profiili (approksimaatio):
      - Alapinta ≈ 0, kun x/c ≥ 0.3 (tasainen runkoa vasten)
      - Etuosassa camber kasvaa pehmeästi 0 → y_t (quadratic blend), jotta siirtymä on sulava
    Palauttaa (x_mm, y_upper_mm, y_lower_mm, camber_mm).
    """
    # Kosinijakauma: tiheämpi pisteistus etureunassa
    i = np.arange(npts + 1)
    x = (1 - np.cos(np.pi * i / npts)) / 2.0
    yt = naca00_thickness_half(x, THICKNESS)

    camber = np.zeros_like(x)
    aft = x >= 0.3
    camber[aft] = yt[aft]
    front = x < 0.3
    xf = x[front]
    camber[front] = yt[front] * (xf / 0.3) ** 2

    y_upper = camber + yt
    y_lower = camber - yt

    x_mm = x * chord_mm
    return x_mm, y_upper * chord_mm, y_lower * chord_mm, camber * chord_mm

def catmull_rom_to_bezier(points: List[Tuple[float, float]]):
    """Uniform Catmull–Rom → Bézier segmentit [(P0,C1,C2,P3), ...]."""
    if len(points) < 4:
        raise ValueError("Tarvitaan ≥ 4 pistettä splinelle.")
    beziers = []
    pts = [points[0]] + points + [points[-1]]
    for i in range(1, len(pts) - 2):
        p0 = np.array(pts[i - 1], float)
        p1 = np.array(pts[i], float)
        p2 = np.array(pts[i + 1], float)
        p3 = np.array(pts[i + 2], float)
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        beziers.append((tuple(p1), tuple(c1), tuple(c2), tuple(p2)))
    return beziers

def make_svg_path_from_points(points: List[Tuple[float, float]], close: bool = True) -> str:
    """Muodosta sileä SVG-path splinestä."""
    if len(points) < 4:
        d = f"M {points[0][0]:.4f},{points[0][1]:.4f} " + " ".join(f"L {x:.4f},{y:.4f}" for x, y in points[1:])
        return d + (" Z" if close else "")
    beziers = catmull_rom_to_bezier(points)
    d = f"M {beziers[0][0][0]:.4f},{beziers[0][0][1]:.4f} "
    for (p0, c1, c2, p3) in beziers:
        d += f"C {c1[0]:.4f},{c1[1]:.4f} {c2[0]:.4f},{c2[1]:.4f} {p3[0]:.4f},{p3[1]:.4f} "
    return d + ("Z" if close else "")

# 1) Pisteet
x_mm, y_u_mm, y_l_mm, camber_mm = clarky_like_13(CHORD_MM, NPTS)

# 2) Camber-tiedot
camber_percent = (np.max(camber_mm) / CHORD_MM) * 100.0
imax = int(np.argmax(camber_mm))
x_camber = (x_mm[imax] / CHORD_MM)

# 3) Suljettu ääriviiva splinelle: ylä (TE→LE) + ala (LE→TE)
upper_pts = list(zip(x_mm[::-1], y_u_mm[::-1]))
lower_pts = list(zip(x_mm[1:],  y_l_mm[1:]))
outline = upper_pts + lower_pts

# 4) SVG
d_path = make_svg_path_from_points(outline, close=True)
minx, maxx = float(np.min(x_mm)), float(np.max(x_mm))
miny = float(min(np.min(y_l_mm), np.min(y_u_mm)))
maxy = float(max(np.max(y_l_mm), np.max(y_u_mm)))
width, height = maxx - minx, maxy - miny
margin = max(width, height) * 0.05
vb = (minx - margin, miny - margin, width + 2*margin, height + 2*margin)

svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}mm" height="{height:.3f}mm"
     viewBox="{vb[0]:.3f} {vb[1]:.3f} {vb[2]:.3f} {vb[3]:.3f}" version="1.1">
  <title>Clark Y -tyylinen 13% (kordi {CHORD_MM:.1f} mm)</title>
  <!-- Maksimi-camber ≈ {camber_percent:.2f} % @ x/c ≈ {x_camber:.3f} -->
  <g fill="none" stroke="black" stroke-width="0.3">
    <path d="{d_path}"/>
    <line x1="0" y1="0" x2="{CHORD_MM:.1f}" y2="0" stroke="gray" stroke-width="0.15" stroke-dasharray="1,1"/>
  </g>
</svg>'''

with open("clarky13_200mm.svg", "w", encoding="utf-8") as f:
    f.write(svg)

# 5) CSV
df = pd.DataFrame({"x_mm": x_mm, "y_upper_mm": y_u_mm, "y_lower_mm": y_l_mm, "camber_mm": camber_mm})
df.to_csv("clarky13_200mm_coords.csv", index=False)

print(f"SVG: clarky13_200mm.svg")
print(f"CSV: clarky13_200mm_coords.csv")
print(f"CAMBER ≈ {camber_percent:.2f}% @ x/c ≈ {x_camber:.3f}")
