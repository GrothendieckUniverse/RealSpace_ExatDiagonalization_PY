"""Unit tests for the bitwise operations, symmetry machinery, and model building."""

import math
import unittest

from fractions import Fraction

import numpy as np

import realspace_exactdiagonalization_py as ed
from realspace_exactdiagonalization_py.utils.bitwise_operations import (
    bitmask_of_site,
    decode_bit_mask_to_configuration,
    decode_bit_mask_to_configuration_inplace,
    empty_site_for_mask,
    encode_configuration_to_bit_mask,
    filled_site_iter_for_mask,
    is_site_empty,
    is_site_occupied,
    n_occupied_for_mask,
    occupy_site_for_mask,
)


class BitwiseOperationsTest(unittest.TestCase):
    """Bitmask encode/decode/occupy semantics (1-based site indices)."""

    def test_basic_bit_ops(self):
        self.assertEqual(bitmask_of_site(1), 1)
        self.assertEqual(bitmask_of_site(3), 4)
        self.assertTrue(is_site_occupied(0b101, 1))
        self.assertFalse(is_site_occupied(0b101, 2))
        self.assertTrue(is_site_empty(0b101, 2))
        self.assertEqual(occupy_site_for_mask(0b101, 2), 0b111)
        self.assertEqual(empty_site_for_mask(0b111, 2), 0b101)
        self.assertEqual(n_occupied_for_mask(0b10110), 3)

    def test_encode_decode_roundtrip(self):
        m = encode_configuration_to_bit_mask([2, 3, 5])
        self.assertEqual(m, 0b10110)
        self.assertEqual(decode_bit_mask_to_configuration(m, 5), [2, 3, 5])
        occ = [0, 0, 0]
        decode_bit_mask_to_configuration_inplace(occ, m, 5)
        self.assertEqual(occ, [2, 3, 5])
        self.assertEqual(list(filled_site_iter_for_mask(m, 5)), [2, 3, 5])

    def test_iterators(self):
        self.assertEqual(list(filled_site_iter_for_mask(0b10110, 5)), [2, 3, 5])


class SymmetryActionTest(unittest.TestCase):
    """Group actions on occupation masks (boson + fermion signs)."""

    def test_boson_permutation_action(self):
        op = ed.Symmetry_Operation("t", [2, 3, 1])  # cyclic shift
        m, phase = ed.apply_operation_to_mask(
            0b101, op, ed.Particle_Statistics.BOSONIC
        )
        self.assertEqual(m, 0b011)  # sites {1,3} -> {1,2}
        self.assertAlmostEqual(abs(phase - 1.0), 0.0)

    def test_fermion_permutation_sign(self):
        # perm = identity, single-particle mask -> trivial phase
        op = ed.Symmetry_Operation("id", [1, 2, 3])
        _, phase = ed.apply_operation_to_mask(
            0b001, op, ed.Particle_Statistics.FERMIONIC
        )
        self.assertAlmostEqual(phase, 1.0 + 0.0j)
        # swap permutation on two-particle state -> odd permutation sign
        op_swap = ed.Symmetry_Operation("swap", [2, 1, 3])
        _, phase_swap = ed.apply_operation_to_mask(
            0b011, op_swap, ed.Particle_Statistics.FERMIONIC
        )
        self.assertAlmostEqual(phase_swap, -1.0 + 0.0j)

    def test_canonical_representative(self):
        ops = [
            ed.Symmetry_Operation("id", [1, 2, 3, 4]),
            ed.Symmetry_Operation("T", [4, 1, 2, 3]),  # cyclic
        ]
        G = ed.Finite_Symmetry_Group("cyclic", ops, identity_idx=0)
        repr_mask, g_idx, amp = ed.get_canonical_representative(
            0b1000, G, ed.Particle_Statistics.BOSONIC
        )
        # T maps site 4 -> 3, so the orbit {4} has representative {3}
        self.assertEqual(repr_mask, 0b0100)
        self.assertEqual(g_idx, 1)

    def test_translation_group_structure(self):
        from tightbinding_py import initialize_real_space_lattice

        lat = initialize_real_space_lattice(
            sample_size=[2, 3], lattice_name="honeycomb", pbc_indicator=[True, True]
        )
        G = ed.build_translation_group(lat)
        self.assertEqual(len(G.operations), 6)
        self.assertEqual(G.operations[0].label, (0, 0))
        irrep_list = ed.build_translation_irrep_list(G, lat)
        self.assertEqual(len(irrep_list), 6)
        self.assertEqual(irrep_list[0].label, (0, 0))


class ModelBuildingTest(unittest.TestCase):
    """Model builders produce consistent ED data."""

    def test_bosonic_haldane_model(self):
        model = ed.build_zero_flux_bosonic_fci_second_quantized_model(
            sample_size=[2, 3], params=ed.params_DNSheng
        )
        self.assertEqual(model.lattice.n_site, 12)
        self.assertEqual(len(model.bilinear_terms), 144)
        # Hermiticity of the term list: the summed amplitude of every
        # directed pair (i,j) equals the conjugate of the summed amplitude
        # of (j,i).  (Note: a directed pair may occur MULTIPLE times as
        # distinct torus bonds — direct and wrapped — exactly as in Julia.)
        summed: dict[tuple[int, int], complex] = {}
        for (i, j, t) in model.bilinear_terms:
            summed[(i, j)] = summed.get((i, j), 0.0j) + t
        for (i, j), t in summed.items():
            self.assertAlmostEqual(
                abs(t - np.conj(summed.get((j, i), 0.0j))), 0.0, places=12
            )

    def test_fermionic_checkerboard_model(self):
        model = ed.build_zero_flux_fermionic_fci_second_quantized_model(
            sample_size=[3, 4], params=ed.params_Sun_Gu_Katsura_Sarma
        )
        self.assertEqual(model.lattice.n_site, 24)
        self.assertEqual(model.particle_statistics, ed.Particle_Statistics.FERMIONIC)
        self.assertEqual(len(model.density_density_terms), 96)  # V3 = 0

    def test_heisenberg_model(self):
        model = ed.build_heisenberg_second_quantized_model(8, J=1.0)
        self.assertEqual(model.lattice.n_site, 8)
        self.assertEqual(model.particle_statistics, ed.Particle_Statistics.BOSONIC)

    def test_spinful_hubbard_model(self):
        model = ed.build_spinful_hubbard_second_quantized_model([2, 2])
        self.assertEqual(model.lattice.n_site, 8)
        self.assertEqual(len(model.density_density_terms), 4)

    def test_filling_fraction_semantics(self):
        model = ed.build_zero_flux_bosonic_fci_second_quantized_model(
            sample_size=[2, 3], params=ed.params_DNSheng
        )
        G = ed.build_translation_group(model.lattice)
        # 3 bosons on 12 vertices = filling 1/4 (half filling of the band)
        data = ed.build_ed_data(
            model, filling_fraction=Fraction(3, 12), symmetry_group=G
        )
        self.assertEqual(data.n_filled, 3)
        self.assertEqual(data.filling_fraction, Fraction(1, 4))
        with self.assertRaises(AssertionError):
            ed.build_ed_data(
                model, filling_fraction=Fraction(1, 5), symmetry_group=G
            )


class SmallEDRegressionTest(unittest.TestCase):
    """Small-system ED energies checked against the Julia reference values."""

    def test_boson_haldane_23_matrix(self):
        model = ed.build_zero_flux_bosonic_fci_second_quantized_model(
            sample_size=[2, 3], params=ed.params_DNSheng
        )
        G = ed.build_translation_group(model.lattice)
        data = ed.build_ed_data(
            model, filling_fraction=Fraction(3, 12), symmetry_group=G
        )
        ed.ed_scan_at_irrep_matrix((0, 0), data, nev=3)
        vals = data.ed_scan_res[0][0]
        # Julia reference (benchmark CSV): -7.163805363423441 at k=(0,0)
        self.assertAlmostEqual(vals[0], -7.163805363423441, places=9)

    def test_matrix_free_matches_matrix(self):
        model = ed.build_zero_flux_bosonic_fci_second_quantized_model(
            sample_size=[2, 3], params=ed.params_DNSheng
        )
        G = ed.build_translation_group(model.lattice)
        data_m = ed.build_ed_data(
            model, filling_fraction=Fraction(3, 12), symmetry_group=G
        )
        ed.ed_scan_at_irrep_matrix((0, 0), data_m, nev=3)
        vals_matrix = data_m.ed_scan_res[0][0]

        data_mf = ed.build_ed_data(
            ed.build_zero_flux_bosonic_fci_second_quantized_model(
                sample_size=[2, 3], params=ed.params_DNSheng
            ),
            filling_fraction=Fraction(3, 12), symmetry_group=ed.build_translation_group(
                ed.build_zero_flux_bosonic_fci_second_quantized_model(
                    sample_size=[2, 3], params=ed.params_DNSheng
                ).lattice
            ),
        )
        ed.ed_scan_at_irrep_matrixfree((0, 0), data_mf, nev=3)
        vals_mf = data_mf.ed_scan_res[0][0]
        np.testing.assert_allclose(vals_matrix, vals_mf, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
