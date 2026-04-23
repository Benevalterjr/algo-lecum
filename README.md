# algo-lecum

Framework experimental para **decisão em espaço latente** combinando:

- **JEPA (Joint-Embedding Predictive Architecture)** para prever representações futuras.
- **Energy-Based Model (EBM)** para pontuar cenários por consistência estrutural.
- Camada estratégica para transformar score em ação e alocação.

> Status: pesquisa/protótipo. Ótimo para iteração científica; ainda requer validação quantitativa robusta antes de uso real em produção financeira.

---

## 1) Teoria resumida

## JEPA

Dado um estado/contexto \(x_t\), treinamos:

- `Encoder`: \(z_t = f_\theta(x_t)\)
- `Predictor`: \(\hat{z}_{t+1} = g_\phi(z_t)\)

objetivo:

\[
\mathcal{L}_{jepa} = \|\hat{z}_{t+1} - z_{t+1}\|^2
\]

onde \(z_{t+1} = f_\theta(x_{t+1})\).

## Energy-Based Model

Treinamos um `EnergyModel` para dar energia menor a pares plausíveis:

- positivo: \((z_t, z_{t+1}^{real})\)
- negativo: \((z_t, z_{t+1}^{fake})\)

objetivo de margem:

\[
\mathcal{L}_{energy} = \text{mean}(\max(0, E_{pos} - E_{neg} + m))
\]

A decisão usa softmax de `-energy` + gap entre os melhores cenários para medir confiança.

---

## 2) Estrutura do projeto

```text
src/lecum/
  models.py         # Encoder, Predictor, EnergyModel, Narrator
  training.py       # dados sintéticos, seed, treino JEPA/EBM
  analysis.py       # probabilidade/gap/confiança + diretivas
  data.py           # contexto de mercado via yfinance
  strategy.py       # scanner multi-ativo + logging estruturado
  allocation.py     # alocação tiered e soft-allocation
  config.py         # hiperparâmetros e configurações centrais
  logging_utils.py  # logger JSON estruturado
  text_generation.py # geração autoregressiva condicionada

tests/
  test_analysis.py
  test_data.py
  test_training.py
  test_repo_smoke.py
  test_config_logging.py

.github/workflows/ci.yml
pyproject.toml
requirements.txt
```

---

## 3) Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependências principais (versionadas):

- `torch`
- `numpy`
- `pandas`
- `yfinance`
- `pytest`, `ruff`, `black` (qualidade/CI)

---

## 4) Exemplo mínimo de uso

```python
import torch.optim as optim

from lecum.config import ModelConfig, TrainConfig
from lecum.training import (
    build_models,
    generate_synthetic_data,
    set_seed,
    train_energy_epoch,
    train_jepa_epoch,
)

set_seed(42)
model_cfg = ModelConfig(input_dim=64, hidden_dim=128, latent_dim=128)
train_cfg = TrainConfig(lr=1e-3, margin=1.0)

encoder, predictor, energy_model = build_models(model_cfg)
x, y = generate_synthetic_data(num_samples=1024, input_dim=model_cfg.input_dim)

opt_jepa = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=train_cfg.lr)
opt_energy = optim.Adam(energy_model.parameters(), lr=train_cfg.lr)

jepa_loss = train_jepa_epoch(encoder, predictor, x, y, opt_jepa)
energy_loss = train_energy_epoch(encoder, energy_model, x, y, opt_energy, margin=train_cfg.margin)

print({"jepa_loss": jepa_loss, "energy_loss": energy_loss})
```

---

## 5) Configuração centralizada

Use `lecum.config` para controlar hiperparâmetros em um lugar só:

- `ModelConfig`
- `TrainConfig`
- `MarketConfig`
- `AllocationConfig`
- `AppConfig`

Isso reduz hardcode e facilita experimentação reproduzível.

---

## 6) Logging estruturado

O módulo `logging_utils.py` fornece logger JSON para facilitar debug/auditoria:

```python
from lecum.logging_utils import get_logger

logger = get_logger("lecum.strategy")
logger.info("ticker scored", extra={"event": "ticker_scored", "meta": {"ticker": "PETR4.SA"}})
```

---

## 7) Qualidade e CI

Pipeline CI (GitHub Actions):

1. instala dependências
2. roda `ruff check .`
3. roda `black --check .`
4. roda `pytest -q`

Execução local:

```bash
ruff check .
black --check .
pytest -q
```

---

## 8) Observações importantes

- O download de dados usa `auto_adjust=False` explícito no `yfinance` para evitar mudança silenciosa de default.
- Este projeto não substitui backtesting institucional com custos de transação, slippage, validação walk-forward e controles de risco formais.


## 9) Geração de texto estilo LLM (versão leve)

Para adicionar geração de texto **similar ao fluxo de LLMs** (autoregressiva), incluímos `src/lecum/text_generation.py` com:

- `TextGenConfig` para hiperparâmetros do decoder.
- `LatentTextGenerator` (Embedding + GRU + LM head) condicionado por vetor latente.
- `SimpleTokenizer` para prototipação rápida.

Exemplo:

```python
import torch

from lecum.text_generation import LatentTextGenerator, SimpleTokenizer, TextGenConfig

vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "cenário", "bullish", "bearish"]
cfg = TextGenConfig(vocab_size=len(vocab), latent_dim=128, max_len=16)
model = LatentTextGenerator(cfg)
tokenizer = SimpleTokenizer(vocab)

z = torch.randn(1, 128)
out_ids = model.generate(z, max_new_tokens=8, temperature=0.8, top_k=5)
texto = tokenizer.decode(out_ids[0].tolist())
print(texto)
```

> Próximo passo para ficar mais próximo de LLM real: trocar GRU por `TransformerDecoder` causal, usar tokenizer BPE/SentencePiece e treinar com corpus maior + RLHF/finetuning supervisionado.
