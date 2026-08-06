import math
import tempfile
import unittest
from pathlib import Path

from scann_rk4_sj import AsteroidPropagator, MpcorbVerifier
from scann_rk4_sj.mpcorb_parse import format_mpcorb_line


def make_catalog(tmpdir: Path) -> Path:
    path = tmpdir / "MPCORB_test.DAT"
    epoch = "2026-01-01"
    a1, e1 = 2.7670962, 0.0789012
    a2, e2 = 3.1, 0.1
    n1 = 0.9856076686 / (a1 ** 1.5)
    n2 = 0.9856076686 / (a2 ** 1.5)
    lines = [
        format_mpcorb_line("TEST001", 3.34, 0.12, epoch, 20.12346, 73.65432,
                           80.98765, 10.56789, e1, n1, a1),
        format_mpcorb_line("TEST002", 14.5, 0.15, epoch, 40.0, 100.0,
                           200.0, 5.0, e2, n2, a2),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


class TestMpcorbCatalog(unittest.TestCase):
    def test_verify_hit_and_clear(self):
        with tempfile.TemporaryDirectory() as td:
            dat = make_catalog(Path(td))
            verifier = MpcorbVerifier(dat, cache_dir=Path(td) / "cache")
            self.assertEqual(verifier.total_count, 2)

            obs_iso = "2026-04-01T12:00:00"
            res = AsteroidPropagator.predict_single(
                a=2.7670962, e=0.0789012, inc_deg=10.56789,
                Omega_deg=80.98765, w_deg=73.65432, M_deg=20.12346,
                epoch_iso="2026-01-01", obs_iso=obs_iso,
                H=3.34, G=0.12,
            )
            # use a site away from the default to test site propagation
            site = (0.0, 0.0, 0.0)
            res2 = AsteroidPropagator.predict_single(
                a=2.7670962, e=0.0789012, inc_deg=10.56789,
                Omega_deg=80.98765, w_deg=73.65432, M_deg=20.12346,
                epoch_iso="2026-01-01", obs_iso=obs_iso,
                H=3.34, G=0.12,
                site_lon_deg=site[0], site_lat_deg=site[1], site_alt_m=site[2],
            )
            results = verifier.verify_targets([
                {"ra_deg": res2["ra_deg"], "dec_deg": res2["dec_deg"],
                 "time": obs_iso, "site": site, "label": "hit-case"},
                {"ra_deg": res2["ra_deg"] + 5.0, "dec_deg": res2["dec_deg"] + 5.0,
                 "time": obs_iso, "site": site, "label": "clear-case"},
            ])
            hit_item = next(r for r in results if r["label"] == "hit-case")
            clear_item = next(r for r in results if r["label"] == "clear-case")
            self.assertEqual(hit_item["local_asteroid_status"], "match")
            self.assertEqual(
                hit_item["local_asteroid_best_match"]["label"], "TEST001"
            )
            self.assertLess(
                hit_item["local_asteroid_best_match"]["sep_arcsec"], 1.0
            )
            self.assertEqual(clear_item["local_asteroid_status"], "clear")

    def test_cache_reload(self):
        with tempfile.TemporaryDirectory() as td:
            dat = make_catalog(Path(td))
            cache = Path(td) / "cache"
            v1 = MpcorbVerifier(dat, cache_dir=cache)
            v2 = MpcorbVerifier(dat, cache_dir=cache)
            self.assertEqual(v1.total_count, v2.total_count)
            self.assertEqual(v2.total_count, 2)


if __name__ == "__main__":
    unittest.main()
