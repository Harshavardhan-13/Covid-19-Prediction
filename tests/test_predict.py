import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from covid_xray.predict import parse_args


class ParseArgsTests(unittest.TestCase):
    def parse_threshold(self, value: str):
        with patch.object(sys, "argv", ["predict", "image.png", "--threshold", value]):
            return parse_args()

    def test_threshold_accepts_bounds(self):
        self.assertEqual(self.parse_threshold("0").threshold, 0.0)
        self.assertEqual(self.parse_threshold("1").threshold, 1.0)

    def test_threshold_rejects_values_below_zero(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse_threshold("-0.01")

    def test_threshold_rejects_values_above_one(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse_threshold("1.01")

    def test_threshold_rejects_nan(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse_threshold("nan")


if __name__ == "__main__":
    unittest.main()
