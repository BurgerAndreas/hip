import pytest
import torch

from hip.frequency_analysis import (
    massweigh_and_eckartprojection_pair_torch,
    massweigh_and_eckartprojection_torch,
)


@pytest.mark.parametrize("coords", [
    [[0., 0., 0.], [1., 0., 0.], [0.2, 1., 0.]],
    [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
])
def test_pair_projection_matches_independent_projections(coords):
    generator = torch.Generator().manual_seed(42)
    a, b = (torch.randn(9, 9, generator=generator, dtype=torch.float64) for _ in range(2))
    coords = torch.tensor(coords, dtype=torch.float64)
    symbols = ["C", "O", "H"]
    actual = massweigh_and_eckartprojection_pair_torch(a, b, coords, symbols)
    expected = (massweigh_and_eckartprojection_torch(a, coords, symbols),
                massweigh_and_eckartprojection_torch(b, coords, symbols))
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
