# Análise Profunda do Projeto `algo-lecum`

## 1) Visão geral

O projeto demonstra uma pipeline experimental que combina:

- codificação latente (`Encoder`),
- predição no espaço latente (`Predictor`),
- seleção por energia (`EnergyModel`),
- camada narrativa para explicabilidade,
- e uma extensão para scanner de ativos com alocação de portfólio.

A implementação está concentrada no notebook `LECUM.ipynb`, com README ainda muito enxuto.

---

## 2) Diagnóstico técnico por camadas

### 2.1 Estrutura e organização

**Pontos fortes**
- O notebook evolui em camadas lógicas (JEPA básico → energia → confiança → ação → mercado → alocação).
- Boa didática para prototipação e validação rápida de ideias.

**Riscos / limitações**
- Projeto mono-arquivo (notebook único), dificultando manutenção, testes e versionamento sem ruído.
- Ausência de estrutura modular (`src/`, `tests/`, `configs/`) e de ponto de execução reproduzível.
- README atual não descreve objetivo, setup, métricas, limitações nem roadmap.

### 2.2 Modelagem (JEPA + Energy-Based)

**Pontos fortes**
- Arquitetura conceitualmente coerente: encoder + predictor + energy scorer.
- Treino inicial de JEPA mostra convergência de perda (0.0691 → 0.0260).
- Loss contrastiva para energia é introduzida com versão simples e depois margem.

**Riscos / limitações**
- Dados sintéticos iniciais (`Y = X + ruído`) são úteis para toy problem, mas não garantem robustez fora desse regime.
- `Encoder` é reutilizado em fases com domínios estatísticos diferentes (ruído sintético vs retorno financeiro) sem normalização explícita por domínio.
- Não há separação clara entre treino, validação e teste (risco de overfitting e leitura otimista).
- Não há baseline comparativo (ex.: random policy, regressão linear, ranking trivial) para quantificar ganho real.

### 2.3 Camada de confiança e decisão

**Pontos fortes**
- Conversão de energia em probabilidade (softmax de energia negativa) facilita interpretação.
- Uso de “gap” entre melhores cenários como heurística de confiança.
- Camada de ação (`determine_next_action`) cria ponte prática entre score e execução.

**Riscos / limitações**
- Limiares de decisão (ex.: 0.6, 0.4, 0.2, 0.05) são heurísticos e não calibrados estatisticamente.
- Probabilidades derivadas de poucas amostras candidatas aleatórias podem ser instáveis.
- Sem backtesting formal, “alta confiança” não implica melhor retorno esperado real.

### 2.4 Pipeline de mercado

**Pontos fortes**
- Integração com `yfinance` para contexto real.
- Scanner multiativo com ranking por `Energy_Gap`.
- Evolução para alocação de portfólio com versão tiered e versão soft-allocation.

**Riscos / limitações**
- Aviso repetido do `yfinance` sobre `auto_adjust` indica dependência de comportamento default que mudou.
- Features ainda limitadas (retornos e volatilidade curta), sem regime macro, volume, liquidez, spreads etc.
- Candidatos de “futuro” são essencialmente vetores aleatórios, não cenários de mercado condicionais aprendidos.
- O relatório intermediário mostrou exposição total de 110% (antes da alocação profissional), sinalizando risco de alavancagem acidental.

### 2.5 Engenharia de software e MLOps

**Pontos fortes**
- Protótipo funcional de ponta a ponta com execução demonstrável.

**Gaps críticos**
- Sem gerenciamento de dependências reproduzível (`requirements.txt`/`pyproject.toml`).
- Sem testes unitários/integrados.
- Sem logging estruturado, tracking de experimentos, seed global, e validação contínua.
- Sem controles explícitos de risco operacional (slippage, custos, limite por ativo/setor, stop, drawdown).

---

## 3) Evidências observadas no estado atual

- README minimalista, insuficiente para onboarding técnico.
- Queda consistente na JEPA Loss (indicativo de aprendizagem no toy setup).
- Energy loss evolui para valores mais negativos no treino simples.
- Em um cenário, refinamento por reamostragem reduziu probabilidade máxima (~20% para ~10%), mostrando instabilidade de confiança.
- Warnings de `yfinance` aparecem múltiplas vezes sobre mudança de default em `auto_adjust`.
- Scanner detectou “high conviction” para alguns ativos no snapshot executado.
- Lógica de alocação por tiers chegou a 110% de exposição; versão profissional corrige para 85% com 15% caixa.

---

## 4) Classificação de maturidade

### Produto / Pesquisa
- **Pesquisa exploratória:** alta.
- **Pronto para produção:** baixo neste estágio.

### Qualidade técnica
- **Arquitetura conceitual:** boa para POC.
- **Confiabilidade estatística:** moderada para baixa (sem validação robusta).
- **Engenharia de produção:** baixa (falta modularização, testes, CI, rastreabilidade).

---

## 5) Plano de evolução recomendado

### Fase 1 (rápida: 1–2 dias)
1. Modularizar notebook em pacotes:
   - `src/models.py`, `src/data.py`, `src/train.py`, `src/strategy.py`, `src/allocation.py`.
2. Fixar dependências e ambiente (`requirements.txt` ou `pyproject.toml`).
3. Definir `auto_adjust` explicitamente no `yfinance`.
4. Adicionar seed global para reprodutibilidade.
5. Expandir README com objetivo, setup, execução e limitações.

### Fase 2 (curto prazo: 3–7 dias)
1. Criar split temporal (train/validation/test) e backtest walk-forward.
2. Implementar métricas financeiras: retorno anualizado, Sharpe, max drawdown, turnover, hit-rate.
3. Construir baseline(s) comparativos.
4. Calibrar limiares da camada de decisão com dados históricos.

### Fase 3 (médio prazo: 1–3 semanas)
1. Gerador de cenários condicionais (em vez de candidatos puramente aleatórios).
2. Gestão de risco institucional:
   - limites por ativo/setor,
   - custo de transação/slippage,
   - regras de redução de risco por drawdown.
3. Pipeline de experimentos com tracking de runs e versionamento de artefatos.
4. Testes automatizados + CI.

---

## 6) Conclusão executiva

O projeto tem **boa direção conceitual** e demonstra capacidade de transformar um framework latente/energético em decisões operacionais interpretáveis. Como POC, está convincente. Para uso real em mercado, o principal gargalo não é a ideia, mas **maturidade de engenharia + validação quantitativa robusta**. A evolução recomendada é priorizar reprodutibilidade, avaliação estatística e controles de risco antes de escalar complexidade do modelo.
