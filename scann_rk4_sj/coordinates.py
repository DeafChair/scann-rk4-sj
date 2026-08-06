#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""坐标解析与角距离工具（供 CLI match 使用）。"""

from __future__ import annotations

import math
import re


def _parse_sexagesimal_ra(tokens: list[str]) -> float:
    h, m, s = (float(t) for t in tokens[:3])
    return (h + m / 60.0 + s / 3600.0) * 15.0


def _parse_sexagesimal_dec(tokens: list[str]) -> float:
    sign = -1.0 if tokens[0].startswith("-") else 1.0
    d, m, s = (abs(float(t)) for t in tokens[:3])
    return sign * (d + m / 60.0 + s / 3600.0)


def parse_ra_dec(text: str) -> tuple[float, float]:
    """Parse a target coordinate into (ra_deg, dec_deg).

    Accepts:
        "208.845615,-6.191733"            decimal degrees
        "13:55:22.95,-06:11:30.2"         sexagesimal
        "13 55 22.95 -06 11 30.2"         sexagesimal (space separated)
    """
    text = text.strip()
    if not text:
        raise ValueError("empty coordinate")
    tokens = [t.strip() for t in re.split(r"[,;\s]+", text) if t.strip()]
    if len(tokens) == 2:
        if ":" in tokens[0] or ":" in tokens[1]:
            ra_parts = [t for t in re.split(r"[:hHmsS\s]+", tokens[0]) if t]
            dec_parts = [t for t in re.split(r"[:dDmsS\s]+", tokens[1]) if t]
            if len(ra_parts) >= 3 and len(dec_parts) >= 3:
                return _parse_sexagesimal_ra(ra_parts), _parse_sexagesimal_dec(dec_parts)
            raise ValueError(f"cannot parse coordinate: {text!r}")
        return float(tokens[0]), float(tokens[1])
    if len(tokens) == 6:
        return _parse_sexagesimal_ra(tokens[:3]), _parse_sexagesimal_dec(tokens[3:6])
    # colon-separated sexagesimal
    ra_txt, dec_txt = re.split(r"[,;\s]+", text, maxsplit=1)
    ra_parts = [t for t in re.split(r"[:hHmsS\s]+", ra_txt) if t]
    dec_parts = [t for t in re.split(r"[:dDmsS\s]+", dec_txt) if t]
    if len(ra_parts) >= 3 and len(dec_parts) >= 3:
        return _parse_sexagesimal_ra(ra_parts), _parse_sexagesimal_dec(dec_parts)
    raise ValueError(f"cannot parse coordinate: {text!r}")


def separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in arcseconds."""
    dra = math.radians((ra1 - ra2 + 540.0) % 360.0 - 180.0)
    d1 = math.radians(dec1)
    d2 = math.radians(dec2)
    hav = (
        math.sin((d2 - d1) / 2.0) ** 2
        + math.cos(d1) * math.cos(d2) * math.sin(dra / 2.0) ** 2
    )
    return math.degrees(2.0 * math.asin(math.sqrt(min(1.0, hav)))) * 3600.0
