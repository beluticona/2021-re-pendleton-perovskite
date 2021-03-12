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

`pipenv install -e .` install all packeges found by setup. In this case we want src to be importable from notebooks.



