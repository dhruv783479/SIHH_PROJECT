import os
import tempfile
import unittest
from unittest import mock

import onnxruntime as ort

from models.onnx_object_detector import ONNXObjectDetector


class _FakeInput:
    name = "images"
    shape = [1, 3, 640, 640]


class _FakeSession:
    def __init__(self, *a, **k):
        pass

    def get_inputs(self):
        return [_FakeInput()]


class TestLoadRetry(unittest.TestCase):
    def _weights(self):
        fh = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        fh.write(b"x")
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_low_memory_retries_same_runtime_before_cv2(self):
        calls = []

        def fake(path, opts, providers=None):
            calls.append(opts.graph_optimization_level)
            if len(calls) == 1:
                raise RuntimeError("bad allocation")
            return _FakeSession()

        with mock.patch.object(ort, "InferenceSession", side_effect=fake):
            det = ONNXObjectDetector(weights_path=self._weights())
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        self.assertTrue(det.backend.startswith("onnxruntime:"))

    def test_both_attempts_fail_falls_back_to_cv2(self):
        with mock.patch.object(ort, "InferenceSession", side_effect=RuntimeError("bad allocation")), \
             mock.patch("cv2.dnn.readNetFromONNX", return_value=object()):
            det = ONNXObjectDetector(weights_path=self._weights())
        self.assertTrue(det.backend.startswith("opencv_dnn:"))


if __name__ == "__main__":
    unittest.main()
