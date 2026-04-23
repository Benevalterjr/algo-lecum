import pytest

torch = pytest.importorskip("torch")


from lecum.analysis import determine_next_action, get_advanced_analysis


def test_advanced_analysis_shapes_and_confidence():
    probs, gap, conf = get_advanced_analysis([-0.5, -0.1, 0.2])
    assert isinstance(probs, torch.Tensor)
    assert probs.shape[0] == 3
    assert abs(float(probs.sum().item()) - 1.0) < 1e-6
    assert gap >= 0
    assert conf in {"HIGH", "LOW"}


def test_determine_next_action_paths():
    assert "EXECUTE" in determine_next_action(0.8, 0.4)
    assert "RE-SAMPLE" in determine_next_action(0.5, 0.01)
    assert "EXPAND CONTEXT" in determine_next_action(0.2, 0.2)
