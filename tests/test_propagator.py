import datetime
import re
import unittest

import numpy as np

from scann_rk4_sj import AsteroidPropagator


ELEMENTS = dict(
    a=2.7670962,
    e=0.0789012,
    inc_deg=10.56789,
    Omega_deg=80.98765,
    w_deg=73.65432,
    M_deg=20.12346,
)


def state_at(prop_class, dt_days, perturbers=(), step=0.5):
    r0, v0 = prop_class.elements_to_state_helio_ecl_j2000(
        ELEMENTS["a"], ELEMENTS["e"], ELEMENTS["inc_deg"],
        ELEMENTS["Omega_deg"], ELEMENTS["w_deg"], ELEMENTS["M_deg"],
    )
    r0e = prop_class.vec_ecl_to_equ(r0)
    v0e = prop_class.vec_ecl_to_equ(v0)
    epoch = prop_class.ordinal_to_time(datetime.date(2026, 1, 1).toordinal())
    return prop_class.propagate_state_rk4_equ(
        r0e, v0e, dt_days, epoch_t=epoch, step_days=step, perturbers=perturbers,
    )


class TestPropagator(unittest.TestCase):
    def test_two_body_energy_drift(self):
        mu = AsteroidPropagator.MU_SUN_AU3_PER_DAY2
        r, v = state_at(AsteroidPropagator, 365.0, perturbers=())
        e0 = 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)
        r1, v1 = AsteroidPropagator.elements_to_state_helio_ecl_j2000(
            ELEMENTS["a"], ELEMENTS["e"], ELEMENTS["inc_deg"],
            ELEMENTS["Omega_deg"], ELEMENTS["w_deg"], ELEMENTS["M_deg"],
        )
        e1 = 0.5 * np.dot(v1, v1) - mu / np.linalg.norm(r1)
        # Energy should be conserved to well below 1e-4 relative.
        self.assertLess(abs(e1 - e0) / abs(e1), 1e-4)

    def test_round_trip(self):
        r, v = state_at(AsteroidPropagator, 120.0, perturbers=AsteroidPropagator.REFINED_PERTURBERS)
        epoch = AsteroidPropagator.ordinal_to_time(datetime.date(2026, 1, 1).toordinal())
        r2, v2 = AsteroidPropagator.propagate_state_rk4_equ(
            r, v, -120.0, epoch_t=epoch, step_days=0.5,
            perturbers=AsteroidPropagator.REFINED_PERTURBERS,
        )
        r0, v0 = AsteroidPropagator.elements_to_state_helio_ecl_j2000(
            ELEMENTS["a"], ELEMENTS["e"], ELEMENTS["inc_deg"],
            ELEMENTS["Omega_deg"], ELEMENTS["w_deg"], ELEMENTS["M_deg"],
        )
        r0e = AsteroidPropagator.vec_ecl_to_equ(r0)
        self.assertLess(np.linalg.norm(r2 - r0e), 1e-4)

    def test_planets_change_orbit(self):
        r_two, _ = state_at(AsteroidPropagator, 365.0, perturbers=())
        r_pert, _ = state_at(
            AsteroidPropagator, 365.0, perturbers=AsteroidPropagator.REFINED_PERTURBERS
        )
        sep = np.linalg.norm(r_two - r_pert)
        self.assertGreater(sep, 1e-4)  # Jupiter+Saturn must move the orbit

    def test_light_time_finite(self):
        epoch_ord = datetime.date(2026, 1, 1).toordinal()
        ra, dec, mag, r, delta = AsteroidPropagator.predict_refined(
            3.34, 0.12, epoch_ord,
            ELEMENTS["M_deg"], ELEMENTS["w_deg"], ELEMENTS["Omega_deg"],
            ELEMENTS["inc_deg"], ELEMENTS["e"], 0.0, ELEMENTS["a"],
            "2026-08-06T18:00:00",
        )
        self.assertTrue(np.all(np.isfinite(ra)))
        self.assertTrue(np.all(np.isfinite(dec)))
        self.assertGreater(delta[0], 0.1)  # a few tenths of an AU away
        self.assertLess(delta[0], 5.0)

    @unittest.skipUnless(
        AsteroidPropagator.ASTROPY_AVAILABLE, "astropy not available"
    )
    def test_horizons_comparison(self):
        import json
        import urllib.parse
        import urllib.request

        from astropy.time import Time

        def horizons_post(params):
            data = urllib.parse.urlencode({**params, "format": "json"}).encode("utf-8")
            req = urllib.request.Request(
                "https://ssd.jpl.nasa.gov/api/horizons.api", data=data
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("error"):
                raise RuntimeError(payload["error"]["message"])
            return payload["result"]

        try:
            ele_text = horizons_post({
                "COMMAND": "'Ceres'",
                "EPHEM_TYPE": "ELEMENTS",
                "CENTER": "'500@10'",
                "START_TIME": "'2026-08-01'",
                "STOP_TIME": "'2026-08-02'",
                "STEP_SIZE": "'1d'",
            })
            vec_text = horizons_post({
                "COMMAND": "'Ceres'",
                "EPHEM_TYPE": "VECTORS",
                "CENTER": "'500@10'",
                "START_TIME": "'2026-08-01'",
                "STOP_TIME": "'2026-08-02'",
                "STEP_SIZE": "'1d'",
                "QUANTITIES": "'2'",
                "OUT_UNITS": "'AU-D'",
            })
        except Exception as exc:  # pragma: no cover - network
            self.skipTest(f"Horizons unavailable: {exc}")

        m_elem = re.search(
            r"EC=\s*([-+0-9.E]+).*?IN=\s*([-+0-9.E]+).*?OM=\s*([-+0-9.E]+).*?"
            r"W\s*=\s*([-+0-9.E]+).*?MA=\s*([-+0-9.E]+).*?(?<![A-Z0-9])A\s*=\s*([-+0-9.E]+)",
            ele_text,
            re.S,
        )
        m_jd = re.search(r"^\s*(\d{7}\.\d+)\s+=\s+A\.D\.", ele_text, re.M)
        m_vec = re.search(
            r"^\s*X =\s*([-+0-9.E]+)\s+Y =\s*([-+0-9.E]+)\s+Z =\s*([-+0-9.E]+)\s*\n"
            r"\s*VX=\s*([-+0-9.E]+)\s+VY=\s*([-+0-9.E]+)\s+VZ=\s*([-+0-9.E]+)",
            vec_text,
            re.M,
        )
        self.assertIsNotNone(m_elem, "cannot parse Horizons elements")
        self.assertIsNotNone(m_jd, "cannot parse Horizons epoch")
        self.assertIsNotNone(m_vec, "cannot parse Horizons vectors")

        a = float(m_elem.group(6)) / 1.495978707e8  # km -> AU
        e = float(m_elem.group(1))
        inc = float(m_elem.group(2))
        Omega = float(m_elem.group(3))
        w = float(m_elem.group(4))
        M = float(m_elem.group(5))
        r_h = np.array([float(x) for x in m_vec.groups()[:3]], dtype=np.float64)
        v_h = np.array([float(x) for x in m_vec.groups()[3:]], dtype=np.float64)

        r_our, v_our = AsteroidPropagator.elements_to_state_helio_ecl_j2000(
            a, e, inc, Omega, w, M
        )
        # Horizons VECTORS with CENTER='500@10' defaults to the ecliptic
        # J2000 frame, which is exactly the frame elements_to_state uses.

        self.assertLess(
            np.linalg.norm(r_our - r_h), 1e-5,
            f"position error {np.linalg.norm(r_our - r_h):.3e} AU",
        )
        self.assertLess(
            np.linalg.norm(v_our - v_h), 1e-7,
            f"velocity error {np.linalg.norm(v_our - v_h):.3e} AU/day",
        )


if __name__ == "__main__":
    unittest.main()
