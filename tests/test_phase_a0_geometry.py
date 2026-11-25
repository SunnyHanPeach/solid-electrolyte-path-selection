"""Regression checks for Phase A0 geometry and reference matching."""
from itertools import product
import unittest

import numpy as np
from ase import Atoms

from src.phase_a0_geometry import (
    assign_mapping,
    compare_signatures,
    exchange,
    local_signature,
    minimum_vectors,
    reaction_coordinate,
    split_shells,
)


class GeometryTests(unittest.TestCase):
    def test_skew_cell_uses_true_minimum_image(self):
        cell = np.array([[4., 0, 0], [3.3, 2.1, 0], [0.7, 0.4, 5.]])
        delta = np.array([[3., 1.3, 2.7], [-4.7, 0.5, 1.2]])
        actual = minimum_vectors(delta, cell, [True] * 3)
        shifts = np.array(list(product(range(-3, 4), repeat=3))) @ cell
        expected = np.linalg.norm(delta[:, None] - shifts, axis=-1).min(axis=1)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), expected, atol=1e-10)

    def test_reaction_coordinate_unwraps_and_follows_arc(self):
        positions = np.array([[9., 1, 0], [0., 1, 0], [0., 2, 0], [1., 2, 0]])
        unwrapped, arc, coordinate = reaction_coordinate(positions, np.eye(3) * 10, [True] * 3)
        np.testing.assert_allclose(unwrapped[-1] - unwrapped[0], [2., 1., 0.])
        np.testing.assert_allclose(arc, [0., 1., 2., 3.])
        np.testing.assert_allclose(coordinate, [0., 1/3, 2/3, 1.])
        with self.assertRaisesRegex(ValueError, "stationary"):
            reaction_coordinate(np.zeros((3, 3)), np.eye(3) * 10, [True] * 3)

    def test_geometry_assignment_detects_same_species_reordering(self):
        atoms = Atoms("LiOO", [[0, 0, 0], [0, 2, 0], [0, 4, 0]], cell=[10.] * 3, pbc=True)
        reordered = atoms[[0, 2, 1]]
        match, residuals = assign_mapping(atoms, reordered)
        np.testing.assert_array_equal(match, [0, 2, 1])
        np.testing.assert_allclose(residuals, 0.)
        wrapped = atoms.copy()
        wrapped.positions[0, 0] += 10.
        match, residuals = assign_mapping(atoms, wrapped)
        np.testing.assert_array_equal(match, [0, 1, 2])
        np.testing.assert_allclose(residuals, 0.)

    def test_shell_gap_and_identity_exchange(self):
        first, second = split_shells(np.array([3.7, 2.15, 3.55, 2.05, 4.8]), .35)
        np.testing.assert_array_equal(first, [3, 1])
        np.testing.assert_array_equal(second, [2, 0])
        self.assertEqual(tuple(map(len, split_shells(np.array([]), .35))), (0, 0))
        self.assertEqual(exchange({1, 3}, {3, 6}), ([6], [1]))

    def test_local_fingerprint_uses_shape_and_is_invariant(self):
        xyz = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 8.]])
        cell = np.eye(3) * 20
        ref = local_signature(xyz, ["Li", "O", "O", "O"], 0, cell, [True] * 3, 3.)
        rotation = np.array([[0., 1, 0], [-1., 0, 0], [0., 0, 1]])
        moved = xyz[[1, 2, 0]] @ rotation + 3.
        same = local_signature(moved, ["O", "O", "Li"], 2, cell @ rotation, [True] * 3, 3.)
        self.assertAlmostEqual(compare_signatures(ref, same), 0., places=10)
        xyz[2] = [-2., 0., 0.]
        bent = local_signature(xyz, ["Li", "O", "O", "O"], 0, cell, [True] * 3, 3.)
        self.assertGreater(compare_signatures(ref, bent), 1.)
        other = local_signature(xyz, ["Li", "N", "O", "O"], 0, cell, [True] * 3, 3.)
        self.assertIsNone(compare_signatures(ref, other))


if __name__ == "__main__":
    unittest.main()
