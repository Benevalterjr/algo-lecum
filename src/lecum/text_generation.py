"""Geração de texto condicionada no espaço latente (LLM-like simplificado)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TextGenConfig:
    vocab_size: int
    latent_dim: int = 128
    emb_dim: int = 128
    hidden_dim: int = 256
    max_len: int = 32
    bos_token_id: int = 1
    eos_token_id: int = 2


class SimpleTokenizer:
    """Tokenizer whitespace minimalista para prototipação."""

    def __init__(self, vocab: list[str]):
        self.id_to_token = vocab
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}

    def encode(self, text: str, unk_token: str = "<unk>") -> list[int]:
        unk = self.token_to_id.get(unk_token, 0)
        return [self.token_to_id.get(tok, unk) for tok in text.strip().split()]

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        for idx in token_ids:
            tok = self.id_to_token[idx]
            if skip_special and tok.startswith("<") and tok.endswith(">"):
                continue
            pieces.append(tok)
        return " ".join(pieces)


class LatentTextGenerator(nn.Module):
    """Decoder autoregressivo leve condicionado por vetor latente.

    Não é um LLM completo, mas replica o padrão causal:
    - embedding de tokens,
    - estado inicial vindo de `z_latent`,
    - geração token-a-token até EOS.
    """

    def __init__(self, cfg: TextGenConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.latent_to_hidden = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.gru = nn.GRU(input_size=cfg.emb_dim, hidden_size=cfg.hidden_dim, batch_first=True)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, z_latent: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher forcing.

        Args:
            z_latent: [B, latent_dim]
            input_ids: [B, T]
        Returns:
            logits: [B, T, vocab_size]
        """
        h0 = self.latent_to_hidden(z_latent).unsqueeze(0)  # [1, B, H]
        x = self.token_emb(input_ids)  # [B, T, E]
        out, _ = self.gru(x, h0)
        logits = self.lm_head(out)
        return logits

    @torch.no_grad()
    def generate(
        self,
        z_latent: torch.Tensor,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = 20,
    ) -> torch.Tensor:
        """Gera sequência autoregressiva com sampling opcional top-k."""
        self.eval()
        batch_size = z_latent.size(0)
        steps = max_new_tokens or self.cfg.max_len

        cur_token = torch.full(
            (batch_size, 1),
            fill_value=self.cfg.bos_token_id,
            dtype=torch.long,
            device=z_latent.device,
        )

        h = self.latent_to_hidden(z_latent).unsqueeze(0)
        generated = [cur_token]

        for _ in range(steps):
            x = self.token_emb(cur_token)
            out, h = self.gru(x, h)
            logits = self.lm_head(out[:, -1, :])

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k is not None and top_k > 0:
                    top_vals, top_idx = torch.topk(scaled, k=min(top_k, scaled.size(-1)), dim=-1)
                    probs = torch.softmax(top_vals, dim=-1)
                    sampled_local = torch.multinomial(probs, num_samples=1)
                    next_token = torch.gather(top_idx, dim=-1, index=sampled_local)
                else:
                    probs = torch.softmax(scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)
            cur_token = next_token

            if torch.all(next_token.squeeze(-1) == self.cfg.eos_token_id):
                break

        return torch.cat(generated, dim=1)
