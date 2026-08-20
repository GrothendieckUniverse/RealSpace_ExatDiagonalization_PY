"""Pre-built models for symmetry-resolved ED (ports of the Julia tests/examples)."""

from .bosonic_fci import (
    build_bosonic_fci_ed_data,
    build_zero_flux_bosonic_fci_second_quantized_model,
    default_fci_sectors,
    params_DNSheng,
    test_bosonic_fci_charge_pump,
    test_bosonic_fci_spectrum_flow,
)
from .fermionic_fci import (
    CHECKERBOARD_TPP_DENOMINATOR,
    build_fermionic_checkerboard_tb_model,
    build_fermionic_fci_ed_data,
    build_phase_explore_fermionic_checkerboard_model,
    build_zero_flux_fermionic_fci_second_quantized_model,
    default_fci_sectors_fermionic,
    my_optimal_param,
    params_Sun_Gu_Katsura_Sarma,
    test_fermionic_fci_charge_pump,
    test_fermionic_fci_full_ed,
    test_fermionic_fci_spectrum_flow,
)
from .heisenberg import (
    build_heisenberg_ed_data,
    build_heisenberg_second_quantized_model,
)
from .hubbard import (
    build_spinful_hubbard_ed_data,
    build_spinful_hubbard_second_quantized_model,
)

__all__ = [
    "params_DNSheng",
    "build_zero_flux_bosonic_fci_second_quantized_model",
    "build_bosonic_fci_ed_data",
    "default_fci_sectors",
    "test_bosonic_fci_spectrum_flow",
    "test_bosonic_fci_charge_pump",
    "params_Sun_Gu_Katsura_Sarma",
    "my_optimal_param",
    "CHECKERBOARD_TPP_DENOMINATOR",
    "build_fermionic_checkerboard_tb_model",
    "build_zero_flux_fermionic_fci_second_quantized_model",
    "build_phase_explore_fermionic_checkerboard_model",
    "build_fermionic_fci_ed_data",
    "default_fci_sectors_fermionic",
    "test_fermionic_fci_full_ed",
    "test_fermionic_fci_spectrum_flow",
    "test_fermionic_fci_charge_pump",
    "build_heisenberg_second_quantized_model",
    "build_heisenberg_ed_data",
    "build_spinful_hubbard_second_quantized_model",
    "build_spinful_hubbard_ed_data",
]
