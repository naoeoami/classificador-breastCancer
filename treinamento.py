"""Treinamento: compara modelos na base Breast Cancer (UCI #14) e salva o de melhor acurácia."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

COLUNAS = ["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps",
           "deg-malig", "breast", "breast-quad", "irradiat"]

# Atributos ordinais: a ordem das faixas carrega informação
ORDINAIS = {
    "age": ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
    "tumor-size": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
                   "40-44", "45-49", "50-54", "55-59"],
    "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23",
                  "24-26", "27-29", "30-32", "33-35", "36-39"],
    "deg-malig": ["1", "2", "3"],
}
NOMINAIS = ["menopause", "node-caps", "breast", "breast-quad", "irradiat"]


def carregar(caminho="data/breast-cancer.data"):
    # '?' = valor ausente (9 casos em node-caps e breast-quad)
    df = pd.read_csv(caminho, names=COLUNAS, na_values="?", dtype=str)
    return df.drop(columns="Class"), df["Class"]


def preprocessador():
    return ColumnTransformer([
        ("ord", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(categories=list(ORDINAIS.values()))),
            ("esc", StandardScaler()),
        ]), list(ORDINAIS)),
        ("nom", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore")),
        ]), NOMINAIS),
    ])


# Modelo -> grade de hiperparâmetros
MODELOS = {
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"]}),
    "Naive Bayes": (BernoulliNB(), {"alpha": [0.1, 0.5, 1.0, 2.0]}),
    "Regressão Logística": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
    "Árvore de Decisão": (DecisionTreeClassifier(random_state=42),
                          {"max_depth": [2, 3, 4, 5, None], "min_samples_leaf": [1, 5, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
                      {"n_estimators": [100, 300], "max_depth": [3, 5, None], "min_samples_leaf": [1, 5]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                          {"n_estimators": [50, 100], "max_depth": [1, 2, 3], "learning_rate": [0.05, 0.1]}),
    "SVM": (SVC(random_state=42), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    "MLP": (MLPClassifier(max_iter=2000, random_state=42),
            {"hidden_layer_sizes": [(5,), (10,), (10, 5)], "alpha": [0.001, 0.01, 0.1]}),
}


def main():
    X, y = carregar()
    print(f"Base: {X.shape[0]} amostras, {X.shape[1]} atributos")
    print("Distribuição das classes:\n" + y.value_counts().to_string() + "\n")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    resultados = {}
    for nome, (modelo, grade) in MODELOS.items():
        pipe = Pipeline([("prep", preprocessador()), ("clf", modelo)])
        busca = GridSearchCV(pipe, {f"clf__{k}": v for k, v in grade.items()},
                             cv=cv, scoring="accuracy", n_jobs=-1)
        busca.fit(X_tr, y_tr)
        resultados[nome] = busca
        print(f"{nome:22s} acurácia CV = {busca.best_score_:.4f}  {busca.best_params_}")

    baseline = y_tr.value_counts(normalize=True).max()
    print(f"\nBaseline (classe majoritária): {baseline:.4f}")

    # Seleção pela acurácia de validação cruzada; o teste fica só para a avaliação final
    melhor_nome = max(resultados, key=lambda n: resultados[n].best_score_)
    melhor = resultados[melhor_nome].best_estimator_
    pred = melhor.predict(X_te)
    print(f"\nMelhor modelo: {melhor_nome}")
    print(f"Acurácia no conjunto de teste: {accuracy_score(y_te, pred):.4f}")
    print(classification_report(y_te, pred))
    print("Matriz de confusão:\n", confusion_matrix(y_te, pred, labels=melhor.classes_), "\n", melhor.classes_)

    # Modelo final retreinado com todos os dados
    melhor.fit(X, y)
    joblib.dump({"nome": melhor_nome, "modelo": melhor}, "modelo.joblib")
    print("\nModelo salvo em modelo.joblib")


if __name__ == "__main__":
    main()
