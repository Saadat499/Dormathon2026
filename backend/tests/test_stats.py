import pytest

from app.stats import clopper_pearson, psi, sample_size


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(8, 10, 0.95)
    assert lo == pytest.approx(0.4439, abs=1e-3) and hi == pytest.approx(0.9748, abs=1e-3)
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0
    assert clopper_pearson(0, 0) == (0.0, 1.0)


def test_sample_size_is_found_and_consistent():
    r = sample_size(0.9, 0.95, 0.95, 0.2)
    n = r["positives_needed"]
    assert n and r["scans_needed"] == pytest.approx(n / 0.2, abs=1)
    x = round(0.95 * n)
    assert clopper_pearson(x, n)[0] >= 0.9
    assert clopper_pearson(round(0.95 * (n - 1)), n - 1)[0] < 0.9 or n == 1


def test_sample_size_edge_cases():
    assert sample_size(0.9, 0.9)["positives_needed"] is None
    assert sample_size(0.9, 0.8)["positives_needed"] is None
    with pytest.raises(ValueError):
        sample_size(1.2, 0.9)
    with pytest.raises(ValueError):
        sample_size(0.9, 0.95, prevalence=0)


def test_psi():
    a = [i / 100 for i in range(100)]
    assert psi(a, a) < 1e-6
    assert psi(a, [x * 0.3 for x in a]) > 0.25
    assert psi([], a) is None
