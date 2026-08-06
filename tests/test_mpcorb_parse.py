import unittest

from scann_rk4_sj.mpcorb_parse import packed_epoch_to_ordinal, parse_mpcorb_line


def build_line():
    chars = [" "] * 200
    chars[0:7] = list("00001  ")
    chars[8:13] = list(f"{3.34:5.2f}")
    chars[14:19] = list(f"{0.12:5.2f}")
    chars[20:25] = list("K261A")  # 2026-01-10
    chars[26:35] = list(f"{20.1234567:9.5f}")
    chars[37:46] = list(f"{73.6543210:9.5f}")
    chars[48:57] = list(f"{80.9876543:9.5f}")
    chars[59:68] = list(f"{10.5678901:9.5f}")
    chars[70:79] = list(f"{0.0789012:9.5f}")
    chars[80:91] = list(f"{0.2142655:11.7f}")
    chars[92:103] = list(f"{2.7670962:11.7f}")
    chars[166:194] = list("Test Asteroid 00001".ljust(28))
    return "".join(chars)


class TestMpcorbParse(unittest.TestCase):
    def test_packed_epoch(self):
        import datetime
        self.assertEqual(packed_epoch_to_ordinal("K261A"), datetime.date(2026, 1, 10).toordinal())

    def test_parse_line(self):
        parsed = parse_mpcorb_line(build_line())
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["label"], "Test Asteroid 00001")
        self.assertAlmostEqual(parsed["H"], 3.34, places=2)
        self.assertAlmostEqual(parsed["G"], 0.12, places=2)
        self.assertAlmostEqual(parsed["M0"], 20.12346, places=4)
        self.assertAlmostEqual(parsed["w"], 73.65432, places=4)
        self.assertAlmostEqual(parsed["Omega"], 80.98765, places=4)
        self.assertAlmostEqual(parsed["inc"], 10.56789, places=4)
        self.assertAlmostEqual(parsed["e"], 0.0789, places=4)
        self.assertAlmostEqual(parsed["n"], 0.2142655, places=6)
        self.assertAlmostEqual(parsed["a"], 2.7670962, places=6)
        self.assertEqual(parsed["epoch_iso"], "2026-01-10")

    def test_short_line_rejected(self):
        self.assertIsNone(parse_mpcorb_line("short"))


if __name__ == "__main__":
    unittest.main()
