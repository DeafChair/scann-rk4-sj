#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RK4 小行星轨道传播器（太阳 + 木星/土星摄动 + 光行时修正）。

从 SCANN 的 MPCORB 核验器（MPCORBLocalVerifier）中提取，物理与原实现一致：

1. 轨道根数（a, e, i, Ω, ω, M）→ 太阳中心黄道 J2000 状态向量；
2. 旋转到赤道 J2000，用 RK4 固定步长积分，加速度 = 太阳 + 木星 + 土星；
3. 行星/地球/太阳位置来自 astropy builtin 星历；
4. 输出 topocentric astrometric RA/Dec（含光行时迭代，未加光行差/章动）。

单位：AU、AU/day、度。时间用 astropy Time（UTC 输入，内部用 TT）。
"""

from __future__ import annotations

import datetime as _dt
from typing import Iterable

import numpy as np

try:
    from astropy.coordinates import (
        EarthLocation,
        GCRS,
        get_body_barycentric,
        solar_system_ephemeris,
    )
    from astropy import units as u
    from astropy.time import Time

    ASTROPY_AVAILABLE = True
except Exception:  # pragma: no cover - astropy is a hard dependency
    ASTROPY_AVAILABLE = False
    EarthLocation = GCRS = get_body_barycentric = solar_system_ephemeris = None
    Time = None
    u = None


class AsteroidPropagator:
    """RK4 + Jupiter/Saturn perturbation propagator with light-time correction."""

    ASTROPY_AVAILABLE = ASTROPY_AVAILABLE
    C_AU_PER_DAY = 173.1446326846693
    GAUSS_K = 0.01720209895
    MU_SUN_AU3_PER_DAY2 = GAUSS_K ** 2
    REFINED_PERTURBERS = ("jupiter", "saturn")
    PLANET_GM_AU3_PER_DAY2 = {
        "jupiter": 2.824760966201868e-07,
        "saturn": 8.459705993376288e-08,
    }
    DEFAULT_SITE_LON_DEG = 87.17905555555556
    DEFAULT_SITE_LAT_DEG = 43.47080555555556
    DEFAULT_SITE_ALT_M = 2066.0
    _planet_pos_cache: dict = {}

    # ------------------------------------------------------------------ time
    @staticmethod
    def ordinal_to_time(epoch_ord: int):
        """Convert an MPC ordinal day to an astropy Time at TT midnight."""
        try:
            date = _dt.date.fromordinal(int(epoch_ord))
            return Time(f"{date.isoformat()} 00:00:00", scale="tt")
        except Exception:
            return Time("J2000", scale="tt")

    @staticmethod
    def obs_day_value(obs_t) -> float:
        """Observation time as ordinal-day float (same convention as epoch_ord)."""
        if not isinstance(obs_t, Time):
            obs_t = Time(obs_t, scale="utc")
        return float(np.asarray(obs_t.tt.jd, dtype=np.float64).item() - 1721424.5)

    @staticmethod
    def as_time(value):
        if isinstance(value, Time):
            return value
        return Time(value, scale="utc")

    # ------------------------------------------------------- coordinate math
    @staticmethod
    def rotmat_ecl_to_equ_j2000() -> np.ndarray:
        eps = np.deg2rad(23.439291111)
        ce = np.cos(eps)
        se = np.sin(eps)
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]],
            dtype=np.float64,
        )

    @classmethod
    def vec_ecl_to_equ(cls, vec) -> np.ndarray:
        rmat = cls.rotmat_ecl_to_equ_j2000()
        vec = np.asarray(vec, dtype=np.float64)
        if vec.ndim == 1 and vec.shape[0] == 3:
            return rmat @ vec
        if vec.ndim == 2 and vec.shape[1] == 3:
            return vec @ rmat.T
        raise ValueError("vec must be shape (3,) or (N,3)")

    # ------------------------------------------------ elements -> state vector
    @classmethod
    def elements_to_state_arrays_helio_ecl_j2000(
        cls, a, e, inc_deg, Omega_deg, w_deg, M_deg
    ) -> tuple[np.ndarray, np.ndarray]:
        """Osculating elements (batch) -> heliocentric ecliptic J2000 (r, v).

        r in AU, v in AU/day. Input angles in degrees.
        """
        a = np.asarray(a, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)
        inc = np.deg2rad(np.asarray(inc_deg, dtype=np.float64))
        Omega = np.deg2rad(np.asarray(Omega_deg, dtype=np.float64))
        w = np.deg2rad(np.asarray(w_deg, dtype=np.float64))
        M = np.deg2rad(np.remainder(np.asarray(M_deg, dtype=np.float64), 360.0))
        a, e, inc, Omega, w, M = np.broadcast_arrays(a, e, inc, Omega, w, M)
        n_obj = a.size
        if n_obj == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        e_clip = np.clip(e, 0.0, 0.999999999)
        E = M.copy()
        for _ in range(20):
            denom = 1.0 - e_clip * np.cos(E)
            denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
            dE = -(E - e_clip * np.sin(E) - M) / denom
            E = E + dE
            if np.max(np.abs(dE)) < 1e-13:
                break

        cosE = np.cos(E)
        sinE = np.sin(E)
        sqrt_1me2 = np.sqrt(np.clip(1.0 - e_clip * e_clip, 0.0, None))
        x_orb = a * (cosE - e_clip)
        y_orb = a * (sqrt_1me2 * sinE)

        n_mean = np.sqrt(cls.MU_SUN_AU3_PER_DAY2 / np.maximum(a * a * a, 1e-24))
        fac = n_mean / np.maximum(1.0 - e_clip * cosE, 1e-14)
        vx_orb = -a * sinE * fac
        vy_orb = a * sqrt_1me2 * cosE * fac

        cosO = np.cos(Omega)
        sinO = np.sin(Omega)
        cosw = np.cos(w)
        sinw = np.sin(w)
        cosi = np.cos(inc)
        sini = np.sin(inc)

        r11 = cosO * cosw - sinO * sinw * cosi
        r12 = -cosO * sinw - sinO * cosw * cosi
        r21 = sinO * cosw + cosO * sinw * cosi
        r22 = -sinO * sinw + cosO * cosw * cosi
        r31 = sinw * sini
        r32 = cosw * sini

        rx = r11 * x_orb + r12 * y_orb
        ry = r21 * x_orb + r22 * y_orb
        rz = r31 * x_orb + r32 * y_orb
        vx = r11 * vx_orb + r12 * vy_orb
        vy = r21 * vx_orb + r22 * vy_orb
        vz = r31 * vx_orb + r32 * vy_orb

        r = np.column_stack([rx.reshape(-1), ry.reshape(-1), rz.reshape(-1)])
        v = np.column_stack([vx.reshape(-1), vy.reshape(-1), vz.reshape(-1)])
        return r, v

    @classmethod
    def elements_to_state_helio_ecl_j2000(
        cls, a, e, inc_deg, Omega_deg, w_deg, M_deg
    ) -> tuple[np.ndarray, np.ndarray]:
        r, v = cls.elements_to_state_arrays_helio_ecl_j2000(
            np.asarray([a]), np.asarray([e]), np.asarray([inc_deg]),
            np.asarray([Omega_deg]), np.asarray([w_deg]), np.asarray([M_deg]),
        )
        return r[0], v[0]

    # ------------------------------------------------------------- dynamics
    @classmethod
    def accel_helio_ecl_j2000(cls, r) -> np.ndarray:
        """Sun-only acceleration in heliocentric ecliptic J2000 (AU/day^2)."""
        r = np.asarray(r, dtype=np.float64)
        if r.ndim == 1:
            rr = float(np.linalg.norm(r))
            if rr <= 0.0:
                return np.zeros(3, dtype=np.float64)
            return -cls.MU_SUN_AU3_PER_DAY2 * r / (rr ** 3)
        if r.ndim == 2 and r.shape[1] == 3:
            rr = np.linalg.norm(r, axis=1)
            rr3 = np.where(rr > 0.0, rr ** 3, np.inf)
            return -cls.MU_SUN_AU3_PER_DAY2 * r / rr3[:, None]
        raise ValueError("r must be shape (3,) or (N,3)")

    @classmethod
    def planet_helio_equ_xyz(cls, body_name: str, t) -> np.ndarray:
        """Heliocentric equatorial J2000 position of a planet (AU), cached."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy is required for planet positions")
        t = cls.as_time(t)
        t_jd = float(t.tt.jd)
        key = (str(body_name).lower(), round(t_jd, 4))
        if key in cls._planet_pos_cache:
            return cls._planet_pos_cache[key]
        t_dyn = t.tt
        with solar_system_ephemeris.set("builtin"):
            body_vec = get_body_barycentric(body_name, t_dyn)
            sun_vec = get_body_barycentric("sun", t_dyn)
        helio = body_vec - sun_vec
        arr = np.stack(
            [
                np.asarray(helio.x.to(u.AU).value, dtype=np.float64),
                np.asarray(helio.y.to(u.AU).value, dtype=np.float64),
                np.asarray(helio.z.to(u.AU).value, dtype=np.float64),
            ],
            axis=-1,
        )
        cls._planet_pos_cache[key] = arr
        return arr

    @classmethod
    def accel_helio_equ_with_planets(cls, r_obj_equ, t, planets: Iterable[str] | None = None) -> np.ndarray:
        """Sun + point-mass planet perturbations in heliocentric equatorial J2000."""
        if planets is None:
            planets = cls.REFINED_PERTURBERS
        r = np.asarray(r_obj_equ, dtype=np.float64)
        single = r.ndim == 1
        if single:
            r = r.reshape(1, 3)
        rr = np.linalg.norm(r, axis=1)
        rr3 = np.where(rr > 0.0, rr ** 3, np.inf)
        acc = -cls.MU_SUN_AU3_PER_DAY2 * r / rr3[:, None]
        for body in planets:
            mu_p = float(cls.PLANET_GM_AU3_PER_DAY2.get(str(body).lower(), 0.0))
            if mu_p <= 0.0:
                continue
            r_p = np.asarray(cls.planet_helio_equ_xyz(body, t), dtype=np.float64)
            if r_p.ndim == 1:
                r_p = np.repeat(r_p.reshape(1, 3), r.shape[0], axis=0)
            elif r_p.shape[0] != r.shape[0]:
                r_p = np.repeat(r_p.reshape(1, 3), r.shape[0], axis=0)
            dr = r_p - r
            d1 = np.linalg.norm(dr, axis=1)
            d2 = np.linalg.norm(r_p, axis=1)
            d1c = np.where(d1 > 0.0, d1 ** 3, np.inf)
            d2c = np.where(d2 > 0.0, d2 ** 3, np.inf)
            acc += mu_p * (dr / d1c[:, None] - r_p / d2c[:, None])
        return acc[0] if single else acc

    @classmethod
    def rk4_step_state_equ(cls, r, v, t, dt_day: float, perturbers: Iterable[str] | None = None):
        dt = float(dt_day)
        r = np.asarray(r, dtype=np.float64).reshape(3)
        v = np.asarray(v, dtype=np.float64).reshape(3)
        k1_r = v
        k1_v = cls.accel_helio_equ_with_planets(r, t, planets=perturbers)
        t2 = t + (0.5 * dt) * u.day
        k2_r = v + 0.5 * dt * k1_v
        k2_v = cls.accel_helio_equ_with_planets(r + 0.5 * dt * k1_r, t2, planets=perturbers)
        k3_r = v + 0.5 * dt * k2_v
        k3_v = cls.accel_helio_equ_with_planets(r + 0.5 * dt * k2_r, t2, planets=perturbers)
        t4 = t + dt * u.day
        k4_r = v + dt * k3_v
        k4_v = cls.accel_helio_equ_with_planets(r + dt * k3_r, t4, planets=perturbers)
        r_new = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        return r_new, v_new

    @classmethod
    def propagate_state_rk4_equ(
        cls,
        r0,
        v0,
        dt_days: float,
        epoch_t,
        step_days: float = 1.0,
        perturbers: Iterable[str] | None = None,
    ):
        r = np.asarray(r0, dtype=np.float64).reshape(3).copy()
        v = np.asarray(v0, dtype=np.float64).reshape(3).copy()
        remain = float(dt_days)
        base = max(float(abs(step_days)), 1e-6)
        if remain == 0.0:
            return r, v
        sign = 1.0 if remain >= 0.0 else -1.0
        h = base * sign
        t_cur = cls.as_time(epoch_t)
        while abs(remain) > 1e-12:
            dt = h if abs(remain) > abs(h) else remain
            r, v = cls.rk4_step_state_equ(r, v, t_cur, dt, perturbers=perturbers)
            t_cur = t_cur + dt * u.day
            remain -= dt
        return r, v

    @classmethod
    def propagate_states_rk4_batch_equ(
        cls,
        r0,
        v0,
        dt_days,
        epoch_t,
        step_days: float = 1.0,
        perturbers: Iterable[str] | None = None,
    ):
        r0 = np.asarray(r0, dtype=np.float64)
        v0 = np.asarray(v0, dtype=np.float64)
        dt_days = np.asarray(dt_days, dtype=np.float64)
        if dt_days.ndim == 0:
            dt_days = np.full((r0.shape[0],), float(dt_days), dtype=np.float64)
        r_out = np.empty_like(r0, dtype=np.float64)
        v_out = np.empty_like(v0, dtype=np.float64)
        for i in range(r0.shape[0]):
            ri, vi = cls.propagate_state_rk4_equ(
                r0[i], v0[i], dt_days[i],
                epoch_t=epoch_t, step_days=step_days, perturbers=perturbers,
            )
            r_out[i] = ri
            v_out[i] = vi
        return r_out, v_out

    # ------------------------------------------------------ observer / astrometry
    @classmethod
    def observer_helio_equ_xyz(cls, obs_t, site_location=None) -> np.ndarray:
        """Observer heliocentric equatorial J2000 vector (AU)."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy is required for observer positions")
        obs_t = cls.as_time(obs_t)
        t_dyn = obs_t.tt
        earth_vec = get_body_barycentric("earth", t_dyn)
        sun_vec = get_body_barycentric("sun", t_dyn)
        helio = earth_vec - sun_vec
        hx = helio.x.to(u.AU).value
        hy = helio.y.to(u.AU).value
        hz = helio.z.to(u.AU).value
        if site_location is not None:
            try:
                site_gcrs = site_location.get_itrs(obstime=obs_t).transform_to(
                    GCRS(obstime=obs_t)
                )
                hx = hx + site_gcrs.cartesian.x.to(u.AU).value
                hy = hy + site_gcrs.cartesian.y.to(u.AU).value
                hz = hz + site_gcrs.cartesian.z.to(u.AU).value
            except Exception:
                pass
        return np.array([hx, hy, hz], dtype=np.float64)

    @classmethod
    def compose_topocentric_astrometric(
        cls,
        obj_equ,
        r_helio,
        H,
        G,
        obs_t,
        site_location=None,
        r_obs_equ_fixed=None,
    ):
        """Topocentric astrometric RA/Dec/mag from object and observer states."""
        obs_t = cls.as_time(obs_t)
        obj_equ = np.asarray(obj_equ, dtype=np.float64)
        r_helio = np.asarray(r_helio, dtype=np.float64)
        H64 = np.asarray(H, dtype=np.float64)
        G64 = np.where(np.isfinite(G), np.asarray(G, dtype=np.float64), 0.15)
        if r_obs_equ_fixed is None:
            r_obs_equ = cls.observer_helio_equ_xyz(obs_t, site_location=site_location)
        else:
            r_obs_equ = np.asarray(r_obs_equ_fixed, dtype=np.float64).reshape(3)
        rho = obj_equ - r_obs_equ.reshape(1, 3)
        gx = rho[:, 0]
        gy = rho[:, 1]
        gz = rho[:, 2]
        delta_topo = np.linalg.norm(rho, axis=1)
        ra = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0
        dec = np.degrees(np.arctan2(gz, np.sqrt(gx * gx + gy * gy)))
        dot = np.sum(obj_equ * rho, axis=1)
        phase = np.degrees(
            np.arccos(
                np.clip(
                    dot / (np.maximum(r_helio, 1e-8) * np.maximum(delta_topo, 1e-8)),
                    -1.0,
                    1.0,
                )
            )
        )
        alpha = np.deg2rad(phase)
        tan_a2 = np.tan(alpha / 2.0)
        phi1 = np.exp(-3.33 * np.power(np.maximum(tan_a2, 0.0), 0.63))
        phi2 = np.exp(-1.87 * np.power(np.maximum(tan_a2, 0.0), 1.22))
        mix = np.maximum((1.0 - G64) * phi1 + G64 * phi2, 1e-12)
        mag = (
            H64
            + 5.0 * np.log10(np.maximum(r_helio, 1e-6) * np.maximum(delta_topo, 1e-6))
            - 2.5 * np.log10(mix)
        )
        return ra, dec, mag, r_helio, delta_topo

    # -------------------------------------------------------------- full predict
    @classmethod
    def propagate_objects_refined_equ(
        cls,
        H,
        G,
        epoch_ord,
        M0,
        w,
        Omega,
        inc,
        e,
        n,
        a,
        emit_t,
        step_days: float = 0.25,
        time_base_mode: str = "per_object_epoch",
    ):
        """Elements -> state at emit_t with sun + Jupiter/Saturn perturbations."""
        if np.asarray(H).size == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        emit_t = cls.as_time(emit_t)
        emit_day = cls.obs_day_value(emit_t)
        emit_ref = emit_t if getattr(emit_t, "isscalar", True) else emit_t[0]
        dt_days = np.atleast_1d(emit_day - np.asarray(epoch_ord, dtype=np.float64))
        r0_ecl, v0_ecl = cls.elements_to_state_arrays_helio_ecl_j2000(
            np.asarray(a, dtype=np.float64),
            np.asarray(e, dtype=np.float64),
            np.asarray(inc, dtype=np.float64),
            np.asarray(Omega, dtype=np.float64),
            np.asarray(w, dtype=np.float64),
            np.asarray(M0, dtype=np.float64),
        )
        r0_equ = cls.vec_ecl_to_equ(r0_ecl)
        v0_equ = cls.vec_ecl_to_equ(v0_ecl)

        if time_base_mode == "per_object_epoch":
            n_obj = r0_equ.shape[0]
            obj_equ = np.empty((n_obj, 3), dtype=np.float64)
            v_equ = np.empty((n_obj, 3), dtype=np.float64)
            epoch_ord_a = np.atleast_1d(np.asarray(epoch_ord, dtype=np.int64))
            for i in range(n_obj):
                epoch_t_i = cls.ordinal_to_time(epoch_ord_a[i])
                ri, vi = cls.propagate_state_rk4_equ(
                    r0_equ[i], v0_equ[i], dt_days[i],
                    epoch_t=epoch_t_i, step_days=step_days,
                    perturbers=cls.REFINED_PERTURBERS,
                )
                obj_equ[i] = ri
                v_equ[i] = vi
        else:
            obj_equ, v_equ = cls.propagate_states_rk4_batch_equ(
                r0_equ, v0_equ, dt_days,
                epoch_t=emit_ref, step_days=step_days,
                perturbers=cls.REFINED_PERTURBERS,
            )
        r_helio = np.linalg.norm(obj_equ, axis=1)
        return obj_equ, v_equ, r_helio

    @classmethod
    def predict_refined(
        cls,
        H,
        G,
        epoch_ord,
        M0,
        w,
        Omega,
        inc,
        e,
        n,
        a,
        obs_t,
        site_location=None,
        step_days: float = 0.5,
        max_iter: int = 4,
        tol_day: float = 1e-8,
        time_base_mode: str = "per_object_epoch",
    ):
        """Topocentric astrometric RA/Dec/mag with light-time iteration."""
        obs_t = cls.as_time(obs_t)
        r_obs_equ_fixed = cls.observer_helio_equ_xyz(obs_t, site_location=site_location)
        H = np.asarray(H, dtype=np.float64)
        G = np.asarray(G, dtype=np.float64)
        epoch_ord = np.asarray(epoch_ord, dtype=np.float64)
        M0 = np.asarray(M0, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        Omega = np.asarray(Omega, dtype=np.float64)
        inc = np.asarray(inc, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)

        obj_equ_0, v_equ_0, r_helio_0 = cls.propagate_objects_refined_equ(
            H, G, epoch_ord, M0, w, Omega, inc, e, n, a,
            obs_t, step_days=step_days, time_base_mode=time_base_mode,
        )
        ra, dec, mag, r, delta = cls.compose_topocentric_astrometric(
            obj_equ_0, r_helio_0, H, G, obs_t,
            site_location=site_location, r_obs_equ_fixed=r_obs_equ_fixed,
        )
        if delta.size == 0:
            return ra, dec, mag, r, delta

        lt_days = np.asarray(delta, dtype=np.float64) / cls.C_AU_PER_DAY
        for _ in range(max(1, int(max_iter))):
            dt_back = -lt_days
            obj_equ_j, v_equ_j = cls.propagate_states_rk4_batch_equ(
                obj_equ_0, v_equ_0, dt_back,
                epoch_t=obs_t, step_days=step_days,
                perturbers=cls.REFINED_PERTURBERS,
            )
            r_helio_j = np.linalg.norm(obj_equ_j, axis=1)
            ra_n, dec_n, mag_n, r_n, delta_n = cls.compose_topocentric_astrometric(
                obj_equ_j, r_helio_j, H, G, obs_t,
                site_location=site_location, r_obs_equ_fixed=r_obs_equ_fixed,
            )
            new_lt = np.asarray(delta_n, dtype=np.float64) / cls.C_AU_PER_DAY
            if np.nanmax(np.abs(new_lt - lt_days)) < float(tol_day):
                return ra_n, dec_n, mag_n, r_n, delta_n
            ra, dec, mag, r, delta = ra_n, dec_n, mag_n, r_n, delta_n
            lt_days = new_lt
        return ra, dec, mag, r, delta

    @classmethod
    def predict_single(
        cls,
        *,
        a: float,
        e: float,
        inc_deg: float,
        Omega_deg: float,
        w_deg: float,
        M_deg: float,
        epoch_iso: str,
        obs_iso: str,
        H: float = 15.0,
        G: float = 0.15,
        site_lon_deg: float | None = None,
        site_lat_deg: float | None = None,
        site_alt_m: float = 0.0,
        step_days: float = 0.5,
        max_iter: int = 4,
    ) -> dict:
        """Convenience wrapper: one object, ISO epoch/observation times."""
        import datetime as _dt2

        epoch_ord = _dt2.date.fromisoformat(epoch_iso).toordinal()
        site = None
        if site_lon_deg is not None and site_lat_deg is not None:
            site = EarthLocation.from_geodetic(
                lon=site_lon_deg * u.deg,
                lat=site_lat_deg * u.deg,
                height=site_alt_m * u.m,
            )
        elif cls.DEFAULT_SITE_LON_DEG and cls.DEFAULT_SITE_LAT_DEG:
            site = EarthLocation.from_geodetic(
                lon=cls.DEFAULT_SITE_LON_DEG * u.deg,
                lat=cls.DEFAULT_SITE_LAT_DEG * u.deg,
                height=cls.DEFAULT_SITE_ALT_M * u.m,
            )
        obs_t = Time(obs_iso, scale="utc")
        ra, dec, mag, r, delta = cls.predict_refined(
            H, G, epoch_ord, M_deg, w_deg, Omega_deg, inc_deg, e, 0.0, a,
            obs_t, site_location=site, step_days=step_days, max_iter=max_iter,
        )
        return {
            "ra_deg": float(ra[0]),
            "dec_deg": float(dec[0]),
            "mag": float(mag[0]),
            "r_au": float(r[0]),
            "delta_au": float(delta[0]),
            "epoch_iso": epoch_iso,
            "obs_iso": obs_iso,
        }
