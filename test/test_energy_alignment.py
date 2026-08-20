"""Energy-alignment regression tests: Python ED vs the Julia reference.

The reference energies below were computed with the Julia package
``RealSpace_ExactDiagonalization.jl`` (system Julia 1.12.7, ARPACK, nev=5,
matrix mode, full translation-group scan) on this machine; the Python port
must reproduce them within ``atol`` (both languages agree to ≲ 2e-13,
well inside ARPACK's convergence tolerance).

Two user-specified test cases:

1. extended Bose–Hubbard model on the Haldane honeycomb lattice in the
   FCI phase (t″ = −0.58) on the [3,4]×2 geometry at half band filling
   (6 hard-core bosons on 24 vertices);
2. extended Fermi–Hubbard model on the checkerboard lattice in the FCI
   phase (t″ = −0.2) on the [3,4]×2 geometry at one-third band filling
   (4 spinless fermions on 24 vertices).
"""

import unittest

from fractions import Fraction

import numpy as np

import realspace_exactdiagonalization_py as ed

ATOL = 1e-9

# Julia reference: (case, k_label) -> lowest-5 eigenvalues per sector
JULIA_REF_BOSON_34 = {
    (0, 0): [
        -14.301456146831018, -13.722130823772558, -13.646661890699379,
        -13.640042308564075, -13.550170758301450,
    ],
    (0, 2): [
        -14.299362590626847, -13.743011957063569, -13.717272063933615,
        -13.586585123115862, -13.554354573641792,
    ],
    (1, 0): [
        -13.766743604919926, -13.699026453227097, -13.584647328593840,
        -13.513897654278470, -13.431386726887309,
    ],
}

JULIA_REF_FERMION_34 = {
    (0, 0): [
        -6.536696664020559, -6.377168957543855, -6.339387043488890,
        -6.192010626043237, -5.495381320129844,
    ],
    (0, 2): [
        -6.832285968463585, -6.488761638231560, -5.835575203799947,
        -5.713457341901553, -5.572292581119391,
    ],
    (1, 0): [
        -6.533812748546839, -6.512852966567995, -6.266923917001573,
        -6.084898163737772, -5.557311700956236,
    ],
}


def _sector_vals(ed_data, k_label):
    """Lowest eigenvalues of one momentum sector from scanned ED data.

    Args:
        ed_data: the ED data structure.
        k_label: the momentum label tuple.

    Returns:
        np.ndarray: the sector eigenvalues.
    """
    for idx, irrep in enumerate(ed_data.irrep_list):
        if irrep.label == k_label:
            return ed_data.ed_scan_res[idx][0]
    raise KeyError(k_label)


class BosonHaldaneAlignmentTest(unittest.TestCase):
    """Case 1: bosonic Haldane honeycomb FCI, [3,4], t'' = -0.58, ν = 1/2 per band."""

    @classmethod
    def setUpClass(cls):
        model = ed.build_zero_flux_bosonic_fci_second_quantized_model(
            sample_size=[3, 4], params=ed.params_DNSheng
        )
        G = ed.build_translation_group(model.lattice)
        cls.ed_data = ed.build_ed_data(
            model, filling_fraction=Fraction(6, 24), symmetry_group=G
        )
        ed.ed_scan(cls.ed_data, nev=5, mode="matrix")

    def test_sector_energies_match_julia(self):
        for k_label, ref in JULIA_REF_BOSON_34.items():
            vals = _sector_vals(self.ed_data, k_label)
            np.testing.assert_allclose(vals[: len(ref)], ref, atol=ATOL)

    def test_ground_state_energy(self):
        vals = _sector_vals(self.ed_data, (0, 0))
        self.assertAlmostEqual(vals[0], -14.301456146831018, places=9)

    def test_two_degenerate_fci_states(self):
        # the two semion FCI ground states live at k=(0,0) and k=(0,2)
        e00 = _sector_vals(self.ed_data, (0, 0))[0]
        e02 = _sector_vals(self.ed_data, (0, 2))[0]
        self.assertLess(e02 - e00, 0.01)


class FermionCheckerboardAlignmentTest(unittest.TestCase):
    """Case 2: fermionic checkerboard FCI, [3,4], t'' = -0.2, ν = 1/3 per band."""

    @classmethod
    def setUpClass(cls):
        params = dict(ed.models.fermionic_fci.my_optimal_param)
        params["t''"] = -0.2
        _, cls.ed_data, cls.n_filled, cls.filling = (
            ed.build_phase_explore_fermionic_checkerboard_model(
                params, [3, 4], Fraction(1, 3)
            )
        )
        ed.ed_scan(cls.ed_data, nev=5, mode="matrix")

    def test_filling(self):
        self.assertEqual(self.n_filled, 4)
        self.assertEqual(self.filling, Fraction(1, 6))

    def test_sector_energies_match_julia(self):
        for k_label, ref in JULIA_REF_FERMION_34.items():
            vals = _sector_vals(self.ed_data, k_label)
            np.testing.assert_allclose(vals[: len(ref)], ref, atol=ATOL)

    def test_ground_state_energy(self):
        vals = _sector_vals(self.ed_data, (0, 2))
        self.assertAlmostEqual(vals[0], -6.832285968463585, places=9)


if __name__ == "__main__":
    unittest.main()
