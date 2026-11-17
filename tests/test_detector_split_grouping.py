"""The train/val/test split must never put near-duplicate photographs on both sides.

training/train_detectors_real.py groups images whose 64-bit dHash differs in <= 4 bits and assigns
each group to one split. The first banding scheme (four 16-bit bands) missed some pairs at exactly
4 bits; this test fails on that scheme and passes on the eight-band one."""
import random
import unittest

try:
    from training import train_detectors_real as t
except Exception as exc:  # cv2/yaml missing in a minimal environment
    t = None
    _why = str(exc)


@unittest.skipIf(t is None, "training module needs cv2/numpy/yaml")
class TestNearDuplicateGrouping(unittest.TestCase):
    def _pairs(self, n, dist, seed):
        rng = random.Random(seed)
        hashes, truth = {}, []
        for i in range(n):
            a = rng.getrandbits(64)
            b = a
            for bit in rng.sample(range(64), dist):
                b ^= 1 << bit
            hashes[f"a{i}"], hashes[f"b{i}"] = a, b
            truth.append((f"a{i}", f"b{i}"))
        return hashes, truth

    def test_every_pair_within_four_bits_is_grouped(self):
        for dist in range(0, 5):
            hashes, truth = self._pairs(1500, dist, seed=dist)
            owner = {k: i for i, c in enumerate(t.cluster_duplicates(hashes, 4)) for k in c}
            missed = [p for p in truth if owner[p[0]] != owner[p[1]]]
            self.assertEqual(missed, [], f"{len(missed)} pairs at distance {dist} were not grouped")

    def test_distant_images_are_not_merged(self):
        hashes, truth = self._pairs(500, 20, seed=99)
        owner = {k: i for i, c in enumerate(t.cluster_duplicates(hashes, 4)) for k in c}
        merged = [p for p in truth if owner[p[0]] == owner[p[1]]]
        self.assertEqual(merged, [])


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(t is None, "training module needs cv2/numpy/yaml")
class TestFrozenTestExclusion(unittest.TestCase):
    def test_copies_and_near_copies_of_the_frozen_set_are_dropped(self):
        fake = {"a.jpg": 0b1010, "near.jpg": 0b1011, "far.jpg": (1 << 40) - 1, "none.jpg": None}
        items = [{"image": k} for k in fake]
        kept, dropped = t.drop_near_duplicates_of(items, [0b1010], max_dist=4, hash_fn=lambda p: fake[p])
        self.assertEqual(dropped, 2)
        self.assertEqual(sorted(i["image"] for i in kept), ["far.jpg", "none.jpg"])

    def test_nothing_is_dropped_without_exclusions(self):
        items = [{"image": "x"}]
        kept, dropped = t.drop_near_duplicates_of(items, [], hash_fn=lambda p: 5)
        self.assertEqual((len(kept), dropped), (1, 0))
