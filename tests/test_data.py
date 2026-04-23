import pytest

torch = pytest.importorskip("torch")


from lecum.data import pad_to_dim


def test_pad_to_dim_pads_correctly():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = pad_to_dim(x, target_dim=6)
    assert y.shape[0] == 6
    assert torch.allclose(y[:3], x)
    assert torch.allclose(y[3:], torch.zeros(3))


def test_pad_to_dim_raises_on_large_input():
    x = torch.ones(10)
    with pytest.raises(ValueError):
        pad_to_dim(x, target_dim=5)
