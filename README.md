# classificador-breastCancer

Classificador para a base [Breast Cancer (UCI #14)](https://archive.ics.uci.edu/dataset/14/breast+cancer): prevê `Class` (`no-recurrence-events` / `recurrence-events`).

## Uso
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python treinamento.py      # treina, compara e salva modelo.joblib
.venv/bin/python inferencia.py --age 40-49 --menopause premeno --tumor-size 20-24 --inv-nodes 0-2 \
    --node-caps no --deg-malig 2 --breast left --breast-quad left_low --irradiat no
.venv/bin/python inferencia.py      # sem argumentos: modo interativo
```

## Treinamento (`treinamento.py`)
1. Carga dos dados (`data/breast-cancer.data`), `?` tratado como ausente.
2. Pré-processamento: imputação pela moda; atributos em faixa (age, tumor-size, inv-nodes, deg-malig) com codificação ordinal + padronização; nominais com one-hot.
3. Divisão treino/teste 75/25 estratificada.
4. GridSearch com validação cruzada estratificada 10-fold para KNN, Naive Bayes, Regressão Logística, Árvore de Decisão, Random Forest, Gradient Boosting, SVM e MLP.
5. Escolha do modelo de maior acurácia na validação cruzada, avaliação no teste (acurácia, relatório, matriz de confusão) e retreino com toda a base.

Resultado atual: KNN (k=15), acurácia CV ≈ 0,743, teste ≈ 0,708 (baseline da classe majoritária ≈ 0,70).
