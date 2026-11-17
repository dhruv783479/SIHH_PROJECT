import os
import tempfile
import unittest

from training import score_by_source as sbs


def _make(root):
    for stem, lbl in (("rdd2022_india_000001", "1 0.5 0.5 0.1 0.1\n3 0.5 0.5 0.2 0.2"),
                      ("rdd2022_india_000002", "0 0.5 0.5 0.1 0.1"),
                      ("rdd2022_japan_000000", "1 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1"),
                      ("andrewmvd__pothole-detection_000007", "3 0.5 0.5 0.3 0.3")):
        os.makedirs(os.path.join(root, "test", "images"), exist_ok=True)
        os.makedirs(os.path.join(root, "test", "labels"), exist_ok=True)
        open(os.path.join(root, "test", "images", stem + ".jpg"), "w").close()
        open(os.path.join(root, "test", "labels", stem + ".txt"), "w").write(lbl)


class BySource(unittest.TestCase):
    def test_source_names_survive_underscores_and_dashes(self):
        self.assertEqual(sbs.source_of("rdd2022_india_000123"), "rdd2022_india")
        self.assertEqual(sbs.source_of("andrewmvd__pothole-detection_000007"), "andrewmvd__pothole-detection")
        self.assertEqual(sbs.source_of("plain"), "plain")

    def test_grouping_and_box_counts(self):
        with tempfile.TemporaryDirectory() as d:
            _make(d)
            g = sbs.group_by_source(d)
            self.assertEqual(sorted(g), ["andrewmvd__pothole-detection", "rdd2022_india", "rdd2022_japan"])
            c = sbs.count_boxes(d)
            self.assertEqual(c["rdd2022_india"]["photographs"], 2)
            self.assertEqual(c["rdd2022_india"]["boxes"]["transverse_crack"], 1)
            self.assertEqual(c["rdd2022_japan"]["boxes"]["transverse_crack"], 2)
            self.assertEqual(c["andrewmvd__pothole-detection"]["boxes"]["pothole"], 1)

    def test_yaml_lists_only_that_sources_photographs(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            _make(d)
            y, n = sbs.write_source_yaml(d, "rdd2022_india", os.path.join(d, "lists"))
            self.assertEqual(n, 2)
            cfg = yaml.safe_load(open(y))
            lines = open(cfg["test"]).read().split()
            self.assertEqual(len(lines), 2)
            self.assertTrue(all("rdd2022_india" in p for p in lines))
            self.assertEqual(cfg["names"][1], "transverse_crack")

    def test_unknown_source_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            _make(d)
            with self.assertRaises(ValueError):
                sbs.write_source_yaml(d, "nope", os.path.join(d, "lists"))


if __name__ == "__main__":
    unittest.main()
