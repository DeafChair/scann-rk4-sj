import unittest

from scann_rk4_sj.coordinates import parse_ra_dec, separation_arcsec


class TestCoordinates(unittest.TestCase):
    def test_decimal(self):
        ra, dec = parse_ra_dec("208.845615,-6.191733")
        self.assertAlmostEqual(ra, 208.845615, places=6)
        self.assertAlmostEqual(dec, -6.191733, places=6)

    def test_colon_sexagesimal(self):
        ra, dec = parse_ra_dec("13:55:22.95,-06:11:30.2")
        self.assertAlmostEqual(ra, 13.9230416667 * 15, places=6)
        self.assertAlmostEqual(dec, -(6 + 11 / 60.0 + 30.2 / 3600.0), places=6)

    def test_space_sexagesimal(self):
        ra, dec = parse_ra_dec("13 55 22.95 -06 11 30.2")
        self.assertAlmostEqual(ra, 13.9230416667 * 15, places=6)
        self.assertAlmostEqual(dec, -(6 + 11 / 60.0 + 30.2 / 3600.0), places=6)

    def test_separation(self):
        # 1 degree separation should be 3600 arcsec
        self.assertAlmostEqual(
            separation_arcsec(10.0, 0.0, 11.0, 0.0), 3600.0, places=6
        )
        self.assertAlmostEqual(separation_arcsec(10.0, 0.0, 10.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
