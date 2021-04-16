# 2021-re-pendleton-perovskite

## Description

Replication of Pendleton et all (2020), a paper about ML understanding of halide perovskite crystal formation without accurate physicochemical features.

Original paper available at:

> Pendleton, I. M. *et al.* Can Machines “Learn” Halide Perovskite Crystal Formation without Accurate Physicochemical Features? *J. Phys. Chem. C* **124**, 13982–13992 (2020).

The main goal of this project is to reproduce results from the original paper, as well as to provide a fully reproducible version on GitHub.

## Reproduction

Run:

`pip install pipenv`

`pipenv --three`

`pipenv install ipykernel`

`pipenv run python -m ipykernel install --user --name=re-pendleton`

Install all packages found by setup. In this case we want src to be importable from notebooks.

`pipenv install -e .`

## TODO

    1) Automatize output for case GBC
        - folder for
        - make_classifier
    2) Enable both intr and extr (now if one is on, the other has to be off)

    3) Merge stratified data and all data(run together)

    4) Create only needed folders, now creating all of them (knn only)

    5) RF features importances

