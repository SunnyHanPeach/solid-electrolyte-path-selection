"""Check peak metrics and reference controls for A2."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from phase_a2_robustness import PATHS, compare_stages, peak_profile, reference_banks


class NoveltyTests(unittest.TestCase):
    def test_known_triangular_excess(self):
        m = peak_profile([0., .4, 1.], [1., 3.4, 2.])
        self.assertAlmostEqual(m["excess_peak_s"], .4)
        self.assertAlmostEqual(m["excess_peak"], 2.)
        self.assertAlmostEqual(m["excess_half_width"], .5)
        self.assertAlmostEqual(m["excess_positive_area"], 1.)
        self.assertEqual(m["central_feature"], 1)

    def test_linear_profile_has_no_excess_peak(self):
        m = peak_profile([0., .5, 1.], [1., 2., 3.])
        self.assertIsNone(m["excess_peak_s"])
        self.assertEqual(m["excess_half_width"], 0.)
        self.assertEqual(m["central_feature"], 0)

    def test_stage_comparison_aligns_distinct_coordinates(self):
        a = np.array([0., .4, 1.])
        b = np.array([0., .5, 1.])
        v = np.array([1., 1.4, 2.])
        w = np.array([1., 1.5, 2.])
        key = ("example", "soap_geometry", "baseline", "raw", "summary")
        rows = compare_stages({(*key,"init"):(a,v,peak_profile(a,v)),
                               (*key,"relaxed"):(b,w,peak_profile(b,w))})
        self.assertAlmostEqual(rows[0]["mean_abs_profile_shift"], 0.)

    def test_reference_counts_distinguish_no_op_and_family_exclusions(self):
        rows = [("A","X","O"),("A","X","O"),("B","Y","S"),("B","Y","S"),("B","Y","S")]
        keys = {p:dict(material_id="new",chemsys="X",anion_family="O") for p in PATHS}
        banks, materials, info = reference_banks(rows,np.ones(len(rows),dtype=bool),keys,"test")
        self.assertEqual(info["quota_per_material"],2)
        for group in banks.values():
            np.testing.assert_array_equal(group["raw"],group["exclude_target_material"])
            np.testing.assert_array_equal(group["exclude_exact_chemistry"],[[2,3,4]])
            np.testing.assert_array_equal(group["exclude_anion_family"],[[2,3,4]])
            self.assertEqual(group["material_balanced"].shape,group["size_matched_unstratified"].shape)
            for draw in group["material_balanced"]:
                np.testing.assert_array_equal(sorted(materials[draw]),["A","A","B","B"])


if __name__=="__main__":
    unittest.main()
