#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MPCORB.DAT 80 列格式解析（字段位置与 SCANN 一致）。

返回的 dict 可直接传给 AsteroidPropagator.predict_refined。
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np


def decode_md(ch: str) -> int | None:
    """Decode the packed month/day character used by MPCORB epochs."""
    if not ch:
        return None
    if ch.isdigit():
        return int(ch)
    table = {
        "A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15,
        "G": 16, "H": 17, "I": 18, "J": 19, "K": 20, "L": 21,
        "M": 22, "N": 23, "O": 24, "P": 25, "Q": 26, "R": 27,
        "S": 28, "T": 29, "U": 30, "V": 31,
    }
    return table.get(ch.upper())


def packed_epoch_to_ordinal(s: str) -> int | None:
    """Convert a packed MPCORB epoch (e.g. 'K26A1') to an ordinal day."""
    s = (s or "").strip()
    if len(s) < 5:
        return None
    base = {"I": 1800, "J": 1900, "K": 2000}.get(s[0].upper())
    if base is None:
        return None
    try:
        year = base + int(s[1:3])
        month = decode_md(s[3])
        day = decode_md(s[4])
        if month is None or day is None:
            return None
        return _dt.date(year, month, day).toordinal()
    except Exception:
        return None


def _safe_float(s: str, default: float = np.nan) -> float:
    s = (s or "").strip()
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def encode_md(value: int) -> str:
    """Encode a month/day (1-31) into the packed MPCORB character."""
    if 1 <= value <= 9:
        return str(value)
    table = {10 + i: ch for i, ch in enumerate(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:22]
    )}
    ch = table.get(value)
    if ch is None:
        raise ValueError(f"cannot pack day/month value {value}")
    return ch


def format_mpcorb_line(
    label: str,
    H: float,
    G: float,
    epoch_iso: str,
    M0: float,
    w: float,
    Omega: float,
    inc: float,
    e: float,
    n: float,
    a: float,
) -> str:
    """Format orbital elements as one MPCORB.DAT 80-column record.

    Mainly useful for building small test catalogs; column layout follows
    the MPC standard (see parse_mpcorb_line).
    """
    date = _dt.date.fromisoformat(epoch_iso)
    year_code = "K" if 2000 <= date.year < 2100 else "J" if 1900 <= date.year < 2000 else "I"
    epoch_packed = (
        f"{year_code}{date.year % 100:02d}"
        f"{encode_md(date.month)}{encode_md(date.day)}"
    )
    chars = [" "] * 200
    chars[0:7] = list(str(label)[:7].ljust(7))
    chars[8:13] = list(f"{H:5.2f}")
    chars[14:19] = list(f"{G:5.2f}")
    chars[20:25] = list(epoch_packed)
    chars[26:35] = list(f"{M0:9.5f}")
    chars[37:46] = list(f"{w:9.5f}")
    chars[48:57] = list(f"{Omega:9.5f}")
    chars[59:68] = list(f"{inc:9.5f}")
    chars[70:79] = list(f"{e:9.5f}")
    chars[80:91] = list(f"{n:11.7f}")
    chars[92:103] = list(f"{a:11.7f}")
    chars[166:194] = list(str(label)[:28].ljust(28))
    return "".join(chars)


def parse_mpcorb_line(line: str) -> dict[str, Any] | None:
    """Parse one MPCORB.DAT line into orbital elements.

    Column layout (MPC standard):
        0:7   packed designation
        8:13  H
        14:19 G
        20:25 packed epoch
        26:35 M
        37:46 argument of perihelion
        48:57 longitude of ascending node
        59:68 inclination
        70:79 eccentricity
        80:91 mean daily motion
        92:103 semi-major axis
        166:194 name
    """
    if len(line) < 103:
        return None
    try:
        packed_id = line[0:7].strip()
        epoch_packed = line[20:25].strip()
        if not packed_id or not epoch_packed:
            return None
        epoch_ord = packed_epoch_to_ordinal(epoch_packed)
        if epoch_ord is None:
            return None
        H = _safe_float(line[8:13], np.nan)
        G = _safe_float(line[14:19], 0.15)
        M0 = _safe_float(line[26:35], np.nan)
        w = _safe_float(line[37:46], np.nan)
        Omega = _safe_float(line[48:57], np.nan)
        inc = _safe_float(line[59:68], np.nan)
        e = _safe_float(line[70:79], np.nan)
        n = _safe_float(line[80:91], np.nan)
        a = _safe_float(line[92:103], np.nan)
        nums = (M0, w, Omega, inc, e, n, a)
        if any(not np.isfinite(v) for v in nums):
            return None
        name = line[166:194].strip() if len(line) >= 194 else ""
        label = name if name else packed_id
        return {
            "label": label,
            "H": float(H),
            "G": float(G if np.isfinite(G) else 0.15),
            "epoch_ord": int(epoch_ord),
            "epoch_iso": _dt.date.fromordinal(int(epoch_ord)).isoformat(),
            "M0": float(M0),
            "w": float(w),
            "Omega": float(Omega),
            "inc": float(inc),
            "e": float(e),
            "n": float(n),
            "a": float(a),
        }
    except Exception:
        return None
