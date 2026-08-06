#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scann-rk4-sj 命令行入口。

用法：
    python -m scann_rk4_sj.cli predict --elements "2.77,0.08,10.6,80.3,73.6,20.0" \
        --epoch 2026-01-01 --time 2026-08-06T18:00:00 --H 3.34 --G 0.12
    python -m scann_rk4_sj.cli predict --mpcorb-line "  ...  "
"""

from __future__ import annotations

import argparse
import json
import sys

from .propagator import AsteroidPropagator
from .mpcorb_parse import parse_mpcorb_line


def _fmt_ra(ra_deg: float) -> str:
    ra = ra_deg / 15.0
    h = int(ra)
    m = int((ra - h) * 60.0)
    s = (ra - h - m / 60.0) * 3600.0
    return f"{h:02d}h {m:02d}m {s:05.2f}s"


def _fmt_dec(dec_deg: float) -> str:
    sign = "+" if dec_deg >= 0 else "-"
    d = abs(dec_deg)
    dd = int(d)
    m = int((d - dd) * 60.0)
    s = (d - dd - m / 60.0) * 3600.0
    return f"{sign}{dd:02d}d {m:02d}m {s:04.1f}s"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RK4 + Jupiter/Saturn asteroid propagation")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="predict topocentric astrometric RA/Dec")
    p.add_argument("--elements", default=None,
                   help='comma list "a,e,inc,Omega,w,M" in degrees/AU')
    p.add_argument("--mpcorb-line", default=None, help="raw MPCORB.DAT line")
    p.add_argument("--epoch", default=None, help="epoch of elements, ISO date (e.g. 2026-01-01)")
    p.add_argument("--time", required=True, help="observation time, ISO (e.g. 2026-08-06T18:00:00)")
    p.add_argument("--H", type=float, default=15.0, help="absolute magnitude")
    p.add_argument("--G", type=float, default=0.15, help="slope parameter")
    p.add_argument("--site", default=None,
                   help='observer "lon,lat,alt_m" (default: Xingming N89)')
    p.add_argument("--step", type=float, default=0.5, help="RK4 step in days")
    p.add_argument("--max-iter", type=int, default=4, help="light-time iterations")
    p.add_argument("--json", action="store_true", help="print JSON")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "predict":
        return 2

    site = None
    if args.site:
        parts = [x.strip() for x in args.site.split(",")]
        if len(parts) != 3:
            print("--site needs lon,lat,alt_m", file=sys.stderr)
            return 2
        site = tuple(float(x) for x in parts)

    if args.mpcorb_line:
        parsed = parse_mpcorb_line(args.mpcorb_line)
        if parsed is None:
            print("cannot parse MPCORB line", file=sys.stderr)
            return 2
        elements = {
            "a": parsed["a"], "e": parsed["e"], "inc_deg": parsed["inc"],
            "Omega_deg": parsed["Omega"], "w_deg": parsed["w"], "M_deg": parsed["M0"],
            "epoch_iso": parsed["epoch_iso"], "H": parsed["H"], "G": parsed["G"],
            "label": parsed["label"],
        }
    elif args.elements and args.epoch:
        vals = [float(x.strip()) for x in args.elements.split(",")]
        if len(vals) != 6:
            print("--elements needs exactly 6 values", file=sys.stderr)
            return 2
        elements = {
            "a": vals[0], "e": vals[1], "inc_deg": vals[2],
            "Omega_deg": vals[3], "w_deg": vals[4], "M_deg": vals[5],
            "epoch_iso": args.epoch, "H": args.H, "G": args.G,
            "label": "custom",
        }
    else:
        print("provide --elements+--epoch or --mpcorb-line", file=sys.stderr)
        return 2

    result = AsteroidPropagator.predict_single(
        a=elements["a"],
        e=elements["e"],
        inc_deg=elements["inc_deg"],
        Omega_deg=elements["Omega_deg"],
        w_deg=elements["w_deg"],
        M_deg=elements["M_deg"],
        epoch_iso=elements["epoch_iso"],
        obs_iso=args.time,
        H=elements["H"],
        G=elements["G"],
        site_lon_deg=site[0] if site else None,
        site_lat_deg=site[1] if site else None,
        site_alt_m=site[2] if site else 0.0,
        step_days=args.step,
        max_iter=args.max_iter,
    )
    result["label"] = elements["label"]

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"{elements['label']}")
        print(f"  RA  {_fmt_ra(result['ra_deg'])}  ({result['ra_deg']:.6f} deg)")
        print(f"  Dec {_fmt_dec(result['dec_deg'])}  ({result['dec_deg']:+.6f} deg)")
        print(f"  mag {result['mag']:.2f}   r={result['r_au']:.4f} AU   "
              f"delta={result['delta_au']:.4f} AU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
