import unittest

import numpy as np

from scripts.evaluate_imu_speedbreaker import (block_bootstrap_f1, block_ids,
                                               guarded_train_mask, prf, reading)


class TestHelpers(unittest.TestCase):
    def test_block_ids_contiguous_and_balanced(self):
        b = block_ids(23, 5)
        self.assertEqual(set(b.tolist()), {0, 1, 2, 3, 4})
        self.assertTrue(np.all(np.diff(b) >= 0))

    def test_guard_removes_neighbours_only(self):
        b = block_ids(20, 4)  # 0..4 | 5..9 | ...
        m = guarded_train_mask(b, 1, guard=1)
        self.assertFalse(m[4])
        self.assertFalse(m[10])
        self.assertFalse(m[5:10].any())
        self.assertTrue(m[3] and m[11])

    def test_prf_known_values(self):
        y = [2, 2, 2, 0, 0]
        p = [2, 0, 2, 2, 0]
        r = prf(y, p, 2)
        self.assertAlmostEqual(r["precision"], 2 / 3)
        self.assertAlmostEqual(r["recall"], 2 / 3)
        self.assertEqual(r["support"], 3)

    def test_prf_class_absent_is_zero_not_error(self):
        self.assertEqual(prf([0, 0], [0, 0], 2)["f1"], 0.0)

    def test_bootstrap_resamples_units_and_is_deterministic(self):
        y = np.array([2] * 6 + [0] * 6)
        p = np.array([2, 2, 2, 0, 0, 0] + [0] * 6)
        units = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 6)
        a = block_bootstrap_f1(y, p, units, 2, reps=200, seed=1)
        b = block_bootstrap_f1(y, p, units, 2, reps=200, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(a[2], 2)
        self.assertLessEqual(a[0], a[1])

    def test_bootstrap_without_class_returns_none(self):
        self.assertEqual(block_bootstrap_f1([0, 0], [0, 0], ["a", "b"], 2, reps=10)[0], None)

    def test_reading_flags(self):
        r = reading("Marked", 0.48, 0.21, 0.61, 1, 60)
        self.assertIn("single_drive", r["flags"])
        self.assertIn("unsettled", r["flags"])
        ok = reading("Smooth", 0.9, 0.85, 0.93, 4, 565)
        self.assertEqual(ok["flags"], [])


if __name__ == "__main__":
    unittest.main()
