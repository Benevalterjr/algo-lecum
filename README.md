# algo-lecum

Toolkit experimental para decisão em espaço latente com abordagem JEPA + Energy-Based modeling, com scanner de ativos e alocação de portfólio.

## O que mudou

O projeto foi modularizado para sair de um notebook monolítico e ganhar base de engenharia:

- pacote Python em `src/lecum/`
- testes em `tests/`
- CI com GitHub Actions em `.github/workflows/ci.yml`
- dependências e build em `pyproject.toml`

## Estrutura

```text
src/lecum/
  models.py       # Encoder, Predictor, EnergyModel, Narrator
  training.py     # dados sintéticos, seed, epochs de treino
  analysis.py     # probabilidade/gap/confiança + diretiva
  data.py         # contexto de mercado (yfinance com auto_adjust explícito)
  strategy.py     # scanner multi-ativo
  allocation.py   # alocação tiered e soft-allocation

tests/
  test_analysis.py
  test_data.py
  test_training.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Executar testes

```bash
pytest -q
```

## Observações importantes

- O acesso a mercado usa `yfinance` com `auto_adjust=False` explícito para evitar ambiguidades de default.
- Esta versão melhora robustez de engenharia, mas **não** substitui um backtest institucional com custos, slippage, validação temporal e governança de risco.
