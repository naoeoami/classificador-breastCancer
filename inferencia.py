"""Inferência: recebe os dados de uma nova paciente e retorna o indicativo de diagnóstico (Class).

Uso:
  python inferencia.py --age 40-49 --menopause premeno --tumor-size 20-24 --inv-nodes 0-2 \
      --node-caps no --deg-malig 2 --breast left --breast-quad left_low --irradiat no
Sem argumentos, pergunta cada atributo interativamente.
"""
import argparse

import joblib
import pandas as pd

from treinamento import ORDINAIS

VALORES = {
    **ORDINAIS,
    "menopause": ["lt40", "ge40", "premeno"],
    "node-caps": ["yes", "no"],
    "breast": ["left", "right"],
    "breast-quad": ["left_up", "left_low", "right_up", "right_low", "central"],
    "irradiat": ["yes", "no"],
}
ATRIBUTOS = ["age", "menopause", "tumor-size", "inv-nodes", "node-caps",
             "deg-malig", "breast", "breast-quad", "irradiat"]


def diagnosticar(paciente, caminho="modelo.joblib"):
    for k in ATRIBUTOS:
        if paciente.get(k) not in VALORES[k]:
            raise ValueError(f"{k}={paciente.get(k)!r} inválido. Opções: {VALORES[k]}")
    salvo = joblib.load(caminho)
    modelo = salvo["modelo"]
    X = pd.DataFrame([paciente], columns=ATRIBUTOS)
    return salvo["nome"], modelo.predict(X)[0]


def main():
    p = argparse.ArgumentParser()
    for k in ATRIBUTOS:
        p.add_argument(f"--{k}", choices=VALORES[k])
    args = {k: getattr(p.parse_args(), k.replace("-", "_")) for k in ATRIBUTOS}
    for k, v in args.items():
        while v not in VALORES[k]:
            v = input(f"{k} {VALORES[k]}: ").strip()
        args[k] = v

    nome, classe = diagnosticar(args)
    print(f"Modelo: {nome}")
    print(f"Indicativo de diagnóstico (Class): {classe}")


if __name__ == "__main__":
    main()
