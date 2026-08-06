#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MPCORB.DAT 全库小行星核验器（SCANN 同款逻辑的独立版）。

用户自行下载 MPCORB.DAT（Minor Planet Center 全量轨道根数文件，约 1.2M 行）。
本模块提供：

1. 解析/缓存：首次解析后保存 .npz，之后秒级加载；
2. 粗筛：两体开普勒解析解 + 光行时，按天区取 shortlist（≤400）；
3. 精细：RK4 + 木星/土星摄动 + 光行时迭代（≤300）；
4. guardrail：精细比粗筛差 >5″ 时回退；
5. 命中判定：与目标 RA/Dec 角距离 < 搜索半径（默认 30″）。

移植自 SCANN 的 MPCORBLocalVerifier，去掉 GUI / SQLite / cv2 依赖。
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .mpcorb_parse import parse_mpcorb_line
from .propagator import AsteroidPropagator
from .coordinates import separation_arcsec

try:
    from astropy.coordinates import EarthLocation
    from astropy import units as u
    from astropy.time import Time
except Exception:  # pragma: no cover
    EarthLocation = None
    u = None
    Time = None


class MpcorbVerifier:
    """Load MPCORB.DAT and verify candidate targets against the catalog."""

    C_AU_PER_DAY = AsteroidPropagator.C_AU_PER_DAY
    SHORTLIST_MAX = 400
    REFINED_TRIGGER_FACTOR = 3.0
    REFINED_MAX_CANDIDATES = 300
    REFINED_STEP_DAYS = 0.5
    REFINED_MAX_ITER = 4
    REFINED_TOL_DAY = 1e-8
    BUILD_CHUNK = 200000
    DEFAULT_SITE = (87.17905555555556, 43.47080555555556, 2066.0)

    def __init__(
        self,
        mpcorb_path: str | Path,
        cache_dir: str | Path | None = None,
        logger: Callable[[str], None] | None = None,
    ):
        self.mpcorb_path = Path(mpcorb_path)
        if not self.mpcorb_path.exists():
            raise FileNotFoundError(f"MPCORB.DAT not found: {self.mpcorb_path}")
        self.cache_dir = Path(cache_dir) if cache_dir else self.mpcorb_path.parent / ".mpcorb_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = logger or (lambda msg: print(f"[MPCORB] {msg}"))
        self._load()
        self._sky_cache_mem: dict = {}

    # ------------------------------------------------------------------ load
    def _cache_key(self) -> str:
        stat = self.mpcorb_path.stat()
        raw = f"{self.mpcorb_path}|{stat.st_size}|{int(stat.st_mtime)}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def _load(self):
        npz_path = self.cache_dir / f"mpcorb_{self._cache_key()}.npz"
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                self.arrays = {
                    k: data[k] for k in (
                        "label", "H", "G", "epoch_ord", "M0", "w",
                        "Omega", "inc", "e", "n", "a",
                    )
                }
                self.total_count = int(self.arrays["H"].size)
                self.log(f"loaded {self.total_count:,} orbits from cache "
                         f"{npz_path.name}")
                return
            except Exception as exc:
                self.log(f"cache invalid ({exc}), re-parsing")
        t0 = time.perf_counter()
        labels, H, G, epoch, M0, w, Omega, inc, e, n, a = [], [], [], [], [], [], [], [], [], [], []
        with open(self.mpcorb_path, encoding="latin-1", errors="replace") as f:
            for line in f:
                p = parse_mpcorb_line(line)
                if p is None:
                    continue
                labels.append(p["label"])
                H.append(p["H"])
                G.append(p["G"])
                epoch.append(p["epoch_ord"])
                M0.append(p["M0"])
                w.append(p["w"])
                Omega.append(p["Omega"])
                inc.append(p["inc"])
                e.append(p["e"])
                n.append(p["n"])
                a.append(p["a"])
        self.arrays = {
            "label": np.asarray(labels, dtype="S48"),
            "H": np.asarray(H, dtype=np.float32),
            "G": np.asarray(G, dtype=np.float32),
            "epoch_ord": np.asarray(epoch, dtype=np.int64),
            "M0": np.asarray(M0, dtype=np.float64),
            "w": np.asarray(w, dtype=np.float64),
            "Omega": np.asarray(Omega, dtype=np.float64),
            "inc": np.asarray(inc, dtype=np.float64),
            "e": np.asarray(e, dtype=np.float64),
            "n": np.asarray(n, dtype=np.float64),
            "a": np.asarray(a, dtype=np.float64),
        }
        self.total_count = int(self.arrays["H"].size)
        try:
            np.savez_compressed(
                npz_path,
                label=self.arrays["label"], H=self.arrays["H"], G=self.arrays["G"],
                epoch_ord=self.arrays["epoch_ord"], M0=self.arrays["M0"],
                w=self.arrays["w"], Omega=self.arrays["Omega"],
                inc=self.arrays["inc"], e=self.arrays["e"],
                n=self.arrays["n"], a=self.arrays["a"],
            )
        except Exception as exc:
            self.log(f"cache write failed: {exc}")
        self.log(f"parsed {self.total_count:,} orbits in "
                 f"{time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _wrap_delta_ra_deg(delta_ra):
        return (np.asarray(delta_ra, dtype=np.float64) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _angular_separation_deg(ra1, dec1, ra2, dec2):
        ra1 = np.asarray(ra1, dtype=np.float64)
        dec1 = np.asarray(dec1, dtype=np.float64)
        dra = np.deg2rad((ra1 - ra2 + 540.0) % 360.0 - 180.0)
        dec1r = np.deg2rad(dec1)
        dec2r = np.deg2rad(dec2)
        hav = (
            np.sin((dec2r - dec1r) / 2.0) ** 2
            + np.cos(dec1r) * np.cos(dec2r) * np.sin(dra / 2.0) ** 2
        )
        return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0))))

    @staticmethod
    def _decode_label(raw) -> str:
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace").strip()
        return str(raw).strip()

    @staticmethod
    def _parse_obs_time(value, exptime_sec=0.0):
        try:
            obs = Time(value, scale="utc")
            if exptime_sec:
                obs = obs - (float(exptime_sec) / 2.0) * u.second
            return obs
        except Exception:
            return None

    @staticmethod
    def _time_bin_floor(obs_t, bin_minutes: int):
        dt = obs_t.utc.datetime
        minute = (dt.minute // bin_minutes) * bin_minutes
        return Time(dt.replace(minute=minute, second=0, microsecond=0), scale="utc")

    @staticmethod
    def _site_from_item(item):
        if item.get("site") is not None:
            lon, lat, alt = item["site"]
        else:
            lon = item.get("site_lon_deg", MpcorbVerifier.DEFAULT_SITE[0])
            lat = item.get("site_lat_deg", MpcorbVerifier.DEFAULT_SITE[1])
            alt = item.get("site_alt_m", MpcorbVerifier.DEFAULT_SITE[2])
        try:
            site = EarthLocation.from_geodetic(
                lon=float(lon) * u.deg, lat=float(lat) * u.deg,
                height=float(alt) * u.m,
            )
        except Exception:
            site = None
        return site, (float(lon), float(lat), float(alt))

    @staticmethod
    def _site_tag(site_meta) -> str:
        return f"{site_meta[0]:.4f},{site_meta[1]:.4f},{site_meta[2]:.0f}"

    # ------------------------------------------------------------- coarse
    def _predict_core(self, H, G, epoch_ord, M0, w, Omega, inc, e, n, a, obs_dt, earth_xyz):
        """Vectorised two-body Kepler prediction (SCANN _predict_core)."""
        H = np.asarray(H, dtype=np.float64)
        if H.size == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty, empty, empty
        obs_day = AsteroidPropagator.obs_day_value(obs_dt)
        dt_days = obs_day - np.asarray(epoch_ord, dtype=np.float64)
        e64 = np.asarray(e, dtype=np.float64)
        a64 = np.asarray(a, dtype=np.float64)
        G64 = np.where(np.isfinite(G), np.asarray(G, dtype=np.float64), 0.15)
        H64 = H
        M = np.deg2rad(np.remainder(
            np.asarray(M0, dtype=np.float64) + np.asarray(n, dtype=np.float64) * dt_days, 360.0
        ))
        E = M.copy()
        for _ in range(12):
            denom = 1.0 - e64 * np.cos(E)
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            dE = -(E - e64 * np.sin(E) - M) / denom
            E += dE
            if np.max(np.abs(dE)) < 1e-10:
                break
        nu = 2.0 * np.arctan2(
            np.sqrt(1.0 + e64) * np.sin(E / 2.0),
            np.sqrt(1.0 - e64) * np.cos(E / 2.0),
        )
        r = a64 * (1.0 - e64 * np.cos(E))
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        w_r = np.deg2rad(np.asarray(w, dtype=np.float64))
        O_r = np.deg2rad(np.asarray(Omega, dtype=np.float64))
        i_r = np.deg2rad(np.asarray(inc, dtype=np.float64))
        cosw, sinw = np.cos(w_r), np.sin(w_r)
        cosO, sinO = np.cos(O_r), np.sin(O_r)
        cosi, sini = np.cos(i_r), np.sin(i_r)
        x1 = cosw * x_orb - sinw * y_orb
        y1 = sinw * x_orb + cosw * y_orb
        y2 = cosi * y1
        z2 = sini * y1
        xe = cosO * x1 - sinO * y2
        ye = sinO * x1 + cosO * y2
        ze = z2
        obj_ecl = np.column_stack([xe, ye, ze])
        obj_equ = AsteroidPropagator.vec_ecl_to_equ(obj_ecl)
        xq = obj_equ[:, 0]
        yq = obj_equ[:, 1]
        zq = obj_equ[:, 2]
        ex, ey, ez = np.asarray(earth_xyz, dtype=np.float64).reshape(3)
        gx = xq - ex
        gy = yq - ey
        gz = zq - ez
        delta = np.sqrt(gx * gx + gy * gy + gz * gz)
        ra = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0
        dec = np.degrees(np.arctan2(gz, np.sqrt(gx * gx + gy * gy)))
        dot = xq * gx + yq * gy + zq * gz
        phase = np.degrees(np.arccos(np.clip(
            dot / (np.maximum(r, 1e-8) * np.maximum(delta, 1e-8)), -1.0, 1.0
        )))
        alpha = np.deg2rad(phase)
        tan_a2 = np.tan(alpha / 2.0)
        phi1 = np.exp(-3.33 * np.power(np.maximum(tan_a2, 0.0), 0.63))
        phi2 = np.exp(-1.87 * np.power(np.maximum(tan_a2, 0.0), 1.22))
        mix = np.maximum((1.0 - G64) * phi1 + G64 * phi2, 1e-12)
        mag = (
            H64
            + 5.0 * np.log10(np.maximum(r, 1e-6) * np.maximum(delta, 1e-6))
            - 2.5 * np.log10(mix)
        )
        return ra, dec, mag, r, delta

    def _predict_indices(self, idxs, obs_t, site_location=None):
        """Legacy two-body prediction with 2 light-time iterations."""
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.size == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty, empty, empty
        earth_xyz = AsteroidPropagator.observer_helio_equ_xyz(obs_t, site_location=site_location)
        args = (
            self.arrays["H"][idxs], self.arrays["G"][idxs],
            self.arrays["epoch_ord"][idxs], self.arrays["M0"][idxs],
            self.arrays["w"][idxs], self.arrays["Omega"][idxs],
            self.arrays["inc"][idxs], self.arrays["e"][idxs],
            self.arrays["n"][idxs], self.arrays["a"][idxs],
        )
        ra, dec, mag, r, delta = self._predict_core(*args, obs_t, earth_xyz)
        if delta.size > 0:
            lt_days = np.asarray(delta, dtype=np.float64) / self.C_AU_PER_DAY
            for _ in range(2):
                emit_t = obs_t - lt_days * u.day
                ra, dec, mag, r, delta = self._predict_core(*args, emit_t, earth_xyz)
                new_lt = np.asarray(delta, dtype=np.float64) / self.C_AU_PER_DAY
                if np.nanmax(np.abs(new_lt - lt_days)) < 1e-7:
                    break
                lt_days = new_lt
        return ra, dec, mag, r, delta

    # ------------------------------------------------------------ sky cache
    def _sky_cache_path(self, bin_iso: str, mag_limit: float, site_tag: str) -> Path:
        safe_iso = bin_iso.replace(":", "").replace(".", "_")
        safe_tag = site_tag.replace(",", "_")
        return self.cache_dir / f"sky_{safe_iso}_m{mag_limit:g}_{safe_tag}.npz"

    def _build_sky_cache(self, bin_dt, mag_limit, site_location=None, site_tag="default"):
        path = self._sky_cache_path(bin_dt.isot, mag_limit, site_tag)
        if path.exists():
            return path
        h_soft = float(mag_limit) + 2.5
        coarse_mag = float(mag_limit) + 1.0
        earth_xyz = AsteroidPropagator.observer_helio_equ_xyz(bin_dt, site_location=site_location)
        out_idx, out_ra, out_dec, out_mag = [], [], [], []
        total = self.total_count
        for start in range(0, total, self.BUILD_CHUNK):
            end = min(total, start + self.BUILD_CHUNK)
            H = self.arrays["H"][start:end]
            mask = np.isfinite(H) & (H <= h_soft)
            if not np.any(mask):
                continue
            idxs = start + np.nonzero(mask)[0]
            ra, dec, mag, _, _ = self._predict_core(
                self.arrays["H"][start:end][mask],
                self.arrays["G"][start:end][mask],
                self.arrays["epoch_ord"][start:end][mask],
                self.arrays["M0"][start:end][mask],
                self.arrays["w"][start:end][mask],
                self.arrays["Omega"][start:end][mask],
                self.arrays["inc"][start:end][mask],
                self.arrays["e"][start:end][mask],
                self.arrays["n"][start:end][mask],
                self.arrays["a"][start:end][mask],
                bin_dt,
                earth_xyz,
            )
            keep = np.isfinite(mag) & (mag <= coarse_mag)
            if np.any(keep):
                out_idx.append(idxs[keep].astype(np.int32, copy=False))
                out_ra.append(ra[keep].astype(np.float32, copy=False))
                out_dec.append(dec[keep].astype(np.float32, copy=False))
                out_mag.append(mag[keep].astype(np.float32, copy=False))
        if out_idx:
            sky = {
                "idx": np.concatenate(out_idx),
                "ra": np.concatenate(out_ra),
                "dec": np.concatenate(out_dec),
                "mag": np.concatenate(out_mag),
            }
        else:
            sky = {
                "idx": np.empty(0, dtype=np.int32),
                "ra": np.empty(0, dtype=np.float32),
                "dec": np.empty(0, dtype=np.float32),
                "mag": np.empty(0, dtype=np.float32),
            }
        try:
            np.savez_compressed(path, **sky, bin_iso=bin_dt.isot, mag_limit=float(mag_limit))
        except Exception as exc:
            self.log(f"sky cache write failed: {exc}")
        self.log(f"sky cache {bin_dt.isot}: {sky['idx'].size:,} bright orbits kept")
        return path

    def _load_sky_cache(self, bin_dt, mag_limit, site_location=None, site_tag="default"):
        key = (bin_dt.isot, float(mag_limit), site_tag)
        if key in self._sky_cache_mem:
            return self._sky_cache_mem[key]
        path = self._build_sky_cache(bin_dt, mag_limit, site_location, site_tag)
        data = np.load(path)
        cache = {
            "idx": data["idx"],
            "ra": data["ra"],
            "dec": data["dec"],
            "mag": data["mag"],
        }
        self._sky_cache_mem = {key: cache}
        return cache

    # -------------------------------------------------------------- verify
    def verify_targets(
        self,
        targets: Iterable[dict[str, Any]],
        search_radius_arcsec: float = 30.0,
        mag_limit: float = 18.5,
        bin_minutes: int = 30,
    ) -> list[dict[str, Any]]:
        search_radius_arcsec = float(search_radius_arcsec)
        mag_limit = float(mag_limit)
        bin_minutes = max(1, int(bin_minutes))
        search_deg = search_radius_arcsec / 3600.0
        coarse_box_deg = max(0.25, search_deg * 80.0)

        prepared = []
        for t in targets:
            item = dict(t)
            try:
                ra0 = float(item["ra_deg"])
                dec0 = float(item["dec_deg"])
            except Exception:
                item.update(local_asteroid_status="error", local_asteroid_error="目标缺少 RA/Dec",
                            local_asteroid_matches=[])
                prepared.append((item, None, None, None, None))
                continue
            obs_dt = self._parse_obs_time(
                item.get("time", item.get("obs_iso")),
                exptime_sec=item.get("exptime_sec", item.get("exposure_sec", 0.0)),
            )
            if obs_dt is None:
                item.update(local_asteroid_status="error", local_asteroid_error="无法解析观测时间",
                            local_asteroid_matches=[])
                prepared.append((item, None, None, None, None))
                continue
            site_location, site_meta = self._site_from_item(item)
            bin_dt = self._time_bin_floor(obs_dt, bin_minutes)
            prepared.append((item, obs_dt, bin_dt, (ra0, dec0), (site_location, site_meta)))

        grouped = {}
        for item, obs_dt, bin_dt, coord, (site_location, site_meta) in prepared:
            if bin_dt is None:
                continue
            tag = self._site_tag(site_meta)
            grouped.setdefault((bin_dt.isot, tag), []).append(
                (item, obs_dt, bin_dt, coord, site_location, site_meta)
            )

        for (bin_iso, site_tag), items in grouped.items():
            _, _, bin_dt, _, site_location, _ = items[0]
            cache = self._load_sky_cache(bin_dt, mag_limit, site_location, site_tag)
            cache_idx = cache["idx"]
            cache_ra = cache["ra"]
            cache_dec = cache["dec"]
            for item, obs_dt, _, coord, site_location, site_meta in items:
                ra0, dec0 = coord
                dra = self._wrap_delta_ra_deg(cache_ra - ra0)
                cos_dec = max(math.cos(math.radians(dec0)), 0.1)
                mask = (
                    (np.abs(cache_dec - dec0) <= coarse_box_deg)
                    & (np.abs(dra) <= coarse_box_deg / cos_dec)
                )
                shortlist = cache_idx[mask]
                if shortlist.size > self.SHORTLIST_MAX:
                    sep0 = self._angular_separation_deg(cache_ra[mask], cache_dec[mask], ra0, dec0)
                    shortlist = shortlist[np.argsort(sep0)[: self.SHORTLIST_MAX]]
                matches = []
                mode_used = "none"
                if shortlist.size:
                    ra_legacy, dec_legacy, mag_legacy, _, _ = self._predict_indices(
                        shortlist, obs_dt, site_location=site_location
                    )
                    sep_legacy = self._angular_separation_deg(ra_legacy, dec_legacy, ra0, dec0)
                    ra_ref = np.full_like(ra_legacy, np.nan)
                    dec_ref = np.full_like(dec_legacy, np.nan)
                    mag_ref = np.full_like(mag_legacy, np.nan)
                    sep_ref = np.full_like(sep_legacy, np.nan)
                    refine_mask = np.isfinite(sep_legacy) & (
                        sep_legacy <= search_deg * self.REFINED_TRIGGER_FACTOR
                    )
                    refine_idx = np.nonzero(refine_mask)[0]
                    if refine_idx.size > self.REFINED_MAX_CANDIDATES:
                        refine_idx = refine_idx[np.argsort(sep_legacy[refine_idx])][
                            : self.REFINED_MAX_CANDIDATES
                        ]
                    if refine_idx.size:
                        ra_sub, dec_sub, mag_sub, _, _ = AsteroidPropagator.predict_refined(
                            self.arrays["H"][shortlist[refine_idx]],
                            self.arrays["G"][shortlist[refine_idx]],
                            self.arrays["epoch_ord"][shortlist[refine_idx]],
                            self.arrays["M0"][shortlist[refine_idx]],
                            self.arrays["w"][shortlist[refine_idx]],
                            self.arrays["Omega"][shortlist[refine_idx]],
                            self.arrays["inc"][shortlist[refine_idx]],
                            self.arrays["e"][shortlist[refine_idx]],
                            self.arrays["n"][shortlist[refine_idx]],
                            self.arrays["a"][shortlist[refine_idx]],
                            obs_dt,
                            site_location=site_location,
                            step_days=self.REFINED_STEP_DAYS,
                            max_iter=self.REFINED_MAX_ITER,
                            tol_day=self.REFINED_TOL_DAY,
                            time_base_mode="per_object_epoch",
                        )
                        ra_ref[refine_idx] = ra_sub
                        dec_ref[refine_idx] = dec_sub
                        mag_ref[refine_idx] = mag_sub
                        sep_ref[refine_idx] = self._angular_separation_deg(
                            ra_sub, dec_sub, ra0, dec0
                        )
                    ref_ok = np.isfinite(sep_ref)
                    leg_ok = np.isfinite(sep_legacy)
                    best_ref = float(np.nanmin(sep_ref[ref_ok])) * 3600.0 if np.any(ref_ok) else np.inf
                    best_leg = float(np.nanmin(sep_legacy[leg_ok])) * 3600.0 if np.any(leg_ok) else np.inf
                    if best_ref <= best_leg + 5.0:
                        hit = np.isfinite(sep_ref) & (sep_ref <= search_deg)
                        use_ref = True
                        mode_used = "refined"
                    else:
                        hit = np.isfinite(sep_legacy) & (sep_legacy <= search_deg)
                        use_ref = False
                        mode_used = "legacy_guardrail"
                    if np.any(hit):
                        hit_idx = np.nonzero(hit)[0]
                        mag_sort = mag_ref if use_ref else mag_legacy
                        sep_sort = sep_ref if use_ref else sep_legacy
                        order = np.lexsort((
                            np.nan_to_num(mag_sort[hit_idx], nan=999.0),
                            sep_sort[hit_idx],
                        ))
                        hit_idx = hit_idx[order][:5]
                        for pos in hit_idx:
                            obj = int(shortlist[pos])
                            matches.append({
                                "label": self._decode_label(self.arrays["label"][obj]),
                                "pred_ra_deg": float(
                                    (ra_ref if use_ref else ra_legacy)[pos]
                                ),
                                "pred_dec_deg": float(
                                    (dec_ref if use_ref else dec_legacy)[pos]
                                ),
                                "pred_mag": float(
                                    (mag_ref if use_ref else mag_legacy)[pos]
                                ),
                                "sep_arcsec": float(
                                    (sep_ref if use_ref else sep_legacy)[pos] * 3600.0
                                ),
                                "prediction_mode": mode_used,
                            })
                item["local_asteroid_matches"] = matches
                item["local_asteroid_best_match"] = matches[0] if matches else None
                item["local_asteroid_status"] = "match" if matches else "clear"
                item["local_asteroid_checked"] = True
                item["site"] = site_meta
        final = []
        for item, *_ in prepared:
            item.setdefault("local_asteroid_checked", True)
            item.setdefault("local_asteroid_matches", [])
            item.setdefault("local_asteroid_status", "error")
            final.append(item)
        return final
