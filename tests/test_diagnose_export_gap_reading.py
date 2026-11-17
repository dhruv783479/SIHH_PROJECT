import importlib.util
import os
import unittest

_P = os.path.join(os.path.dirname(__file__), "..", "colab", "colab_diagnose_export_gap.py")


def _load():
    spec = importlib.util.spec_from_file_location("diag_gap", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # main() is guarded; ultralytics is imported only inside it
    return mod


class TestReading(unittest.TestCase):
    def test_padding_explains_gap(self):
        r = _load().reading(a=0.4005, b=0.3800, c=0.3810)
        self.assertTrue(r["padding_explains_gap"])

    def test_not_explained_when_onnx_differs_from_square_pt(self):
        r = _load().reading(a=0.4005, b=0.3800, c=0.4216)
        self.assertFalse(r["padding_explains_gap"])

    def test_not_explained_when_rect_has_no_effect(self):
        r = _load().reading(a=0.3996, b=0.3990, c=0.3992)
        self.assertFalse(r["padding_explains_gap"])


if __name__ == "__main__":
    unittest.main()
