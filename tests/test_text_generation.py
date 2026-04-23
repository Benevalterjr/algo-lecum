import pytest

torch = pytest.importorskip("torch")

from lecum.text_generation import LatentTextGenerator, SimpleTokenizer, TextGenConfig


def test_text_generator_forward_and_generate_shapes():
    cfg = TextGenConfig(vocab_size=16, latent_dim=8, emb_dim=12, hidden_dim=20, max_len=6)
    model = LatentTextGenerator(cfg)

    z = torch.randn(2, cfg.latent_dim)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 4))

    logits = model(z, input_ids)
    assert logits.shape == (2, 4, cfg.vocab_size)

    out = model.generate(z, max_new_tokens=5, temperature=0.0)
    assert out.shape[0] == 2
    assert out.shape[1] >= 2


def test_simple_tokenizer_encode_decode_roundtrip():
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "mercado", "alta"]
    tok = SimpleTokenizer(vocab)
    ids = tok.encode("mercado alta")
    assert ids == [4, 5]
    text = tok.decode([1, 4, 5, 2])
    assert text == "mercado alta"
