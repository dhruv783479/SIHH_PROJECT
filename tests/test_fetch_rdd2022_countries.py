import os
import tempfile
import unittest
import zipfile

from scripts import fetch_rdd2022_countries as fc


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for n in files:
            z.writestr(n, "x")


class CountryFetch(unittest.TestCase):
    def test_extracts_only_the_requested_countrys_train_folder(self):
        with tempfile.TemporaryDirectory() as d:
            zp = os.path.join(d, "all.zip")
            _zip(zp, ["RDD2022/Japan/train/images/a.jpg", "RDD2022/Japan/train/annotations/xmls/a.xml",
                      "RDD2022/Japan/test/images/t.jpg", "RDD2022/Norway/train/images/n.jpg"])
            out = os.path.join(d, "out")
            fc.extract_country(zp, "Japan", out)
            self.assertTrue(os.path.exists(os.path.join(out, "RDD2022/Japan/train/images/a.jpg")))
            self.assertFalse(os.path.exists(os.path.join(out, "RDD2022/Japan/test/images/t.jpg")))
            self.assertFalse(os.path.exists(os.path.join(out, "RDD2022/Norway")))
            self.assertEqual(fc.country_root("Japan", out), os.path.join(out, "RDD2022", "Japan"))

    def test_missing_country_stops_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            zp = os.path.join(d, "all.zip")
            _zip(zp, ["RDD2022/Norway/train/images/n.jpg"])
            with self.assertRaises(SystemExit):
                fc.extract_country(zp, "Japan", os.path.join(d, "out"))

    def test_nested_country_zip(self):
        with tempfile.TemporaryDirectory() as d:
            inner = os.path.join(d, "RDD2022_Czech.zip")
            _zip(inner, ["Czech/train/images/c.jpg"])
            outer = os.path.join(d, "all.zip")
            with zipfile.ZipFile(outer, "w") as z:
                z.write(inner, "Country_Specific_Data/RDD2022_Czech.zip")
            out = os.path.join(d, "out")
            fc.extract_country(outer, "Czech", out)
            self.assertIsNotNone(fc.country_root("Czech", out))

    def test_unknown_country_is_refused(self):
        with self.assertRaises(SystemExit):
            fc.main(["Atlantis"])


if __name__ == "__main__":
    unittest.main()
