# On the Combinatorial Vulnerability of Access Points in Wi-Fi Fingerprinting

![Methodology for combinatorial AP vulnerability assessment](figures/figure1.png)

Official implementation accompanying:

> **On the Combinatorial Vulnerability of Access Points in Wi-Fi Fingerprinting: Characterization and Efficient Identification**  
> Muhammed Noshin, Muhammad Hannan Akhtar, Mohamed I. AlHajri, and Mohammad Zulkernine (2026)

## Overview

Wi-Fi fingerprint-based indoor localization relies on received signal strength
indicator (RSSI) measurements from distributed access points (APs). Manipulating
a carefully selected group of these measurements can substantially degrade
localization accuracy.

For a given attack budget $k$, this work asks:

> Among $N$ attackable AP units, which subset of $k$ APs produces the
> greatest localization error when manipulated?

We formulate this task as **combinatorial AP vulnerability assessment** and
evaluate neural network (NN) and XGBoost localization models across three public
datasets. Exact enumeration is used where computationally feasible. Genetic
algorithm (GA) and simulated annealing (SA) are evaluated as scalable search
methods, while RSSI-proximity and random sampling provide non-adaptive
baselines.

The results reveal a highly non-uniform vulnerability landscape: most AP
subsets cause moderate degradation, whereas a small fraction produces
substantially larger errors. Average RSSI does not reliably identify these
security-critical combinations. GA and SA recover near-worst-case subsets
without enumerating the complete search space and remain applicable in
configurations containing up to 120 attackable AP units.

## Abstract

Wi-Fi fingerprint-based indoor localization is widely used in Internet of
Things (IoT) environments, yet its dependence on distributed AP infrastructure
creates an important security risk. This work formulates the identification of
the most damaging $k$-AP subset as a combinatorial vulnerability-assessment
problem and studies NN and XGBoost localization models across UJIIndoorLoc,
UTSIndoorLoc, and SODIndoorLoc. Exact enumeration, where feasible, shows that
most AP subsets cause moderate degradation while a small fraction produces
substantially larger localization errors. Selecting APs according to average
RSSI consistently fails to identify these damaging combinations, demonstrating
that AP criticality arises from interactions among signals rather than signal
strength alone. GA and SA provide scalable alternatives that consistently
recover near-worst-case AP subsets without exhaustive enumeration. Both methods
remain effective in configurations containing up to 120 attackable AP units and
reduce assessment time by orders of magnitude relative to exhaustive search.

## Evaluation Scope

| Dataset | Evaluated environment | Attackable AP units | Exact evaluation | Approximate search |
|:--|:--|--:|:--|:--|
| UJIIndoorLoc | Building 2, Floor 3 | 21 | all $k$ | all $k$ |
| UJIIndoorLoc | Building 2, Floor 3 | 53 | $k\leq3$ | $k=1,\ldots,53$ |
| UTSIndoorLoc | Floor 4 | 21 | all $k$ | all $k$ |
| UTSIndoorLoc | Floor 4 | 120 | $k\leq3$ | $k=1,\ldots,120$ |
| SODIndoorLoc | SYL environment | 23 physical APs | all $k$ | all $k$ |

For SODIndoorLoc, the 2.4-GHz and 5-GHz RSSI features associated with the same
physical dual-band AP are grouped and manipulated jointly as one attackable AP
unit.

Complete enumeration is restricted to $k\leq3$ in the expanded
configurations because the number of candidates grows combinatorially. For
example, $\binom{53}{5}=2{,}869{,}685$, while
$\binom{120}{4}=8{,}214{,}570$.

## Methods

- **Brute force** evaluates every feasible subset and provides the exact
  worst-case reference.
- **Genetic algorithm (GA)** evolves fixed-size AP subsets using selection,
  crossover, mutation, and elitism.
- **Simulated annealing (SA)** explores fixed-size subsets through single-swap
  neighbors and probabilistic acceptance.
- **RSSI-proximity** biases subset selection toward APs with stronger average
  RSSI.
- **Random sampling** selects subsets uniformly without localization-error
  feedback.

The vulnerability experiments use frozen NN and XGBoost localization models.
Linear Regression and LightGBM are included in the broader baseline-model
comparison reported in the paper.

## Repository Structure

```text
.
├── data/                         # UJIIndoorLoc inputs used by the core implementation
├── figures/                      # Methodology figure
├── scripts/
│   ├── train_baselines.py
│   ├── run_bruteforce.py
│   ├── run_genetic_algorithm.py
│   ├── run_simulated_annealing.py
│   ├── run_rssi_proximity.py
│   └── run_random_sampling.py
├── search_strategies/
│   ├── algorithms/               # Search implementations
│   ├── models/                   # NN, XGBoost, and linear baselines
│   ├── preprocessing.py
│   └── config.py
└── requirements.txt
```

This public repository contains the core model-training, exact-enumeration, and
search implementations. Internal scripts used only to aggregate repeated runs
or format publication figures are intentionally excluded.

## Setup

```bash
git clone https://github.com/HannanAkhtar/On-the-Combinatorial-Vulnerability-of-Access-Points-in-WiFi-Fingerprinting.git
cd On-the-Combinatorial-Vulnerability-of-Access-Points-in-WiFi-Fingerprinting

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn tensorflow xgboost \
  matplotlib joblib tqdm
```

On Windows, activate the environment with:

```bat
.venv\Scripts\activate
```

## Data

The included command-line implementation expects:

```text
data/TrainingData.csv
data/ValidationData.csv
data/noise.txt
```

Dataset files remain subject to the terms of their original providers. Users
who substitute locally downloaded copies should preserve the expected column
names and update the paths in `search_strategies/config.py` when necessary.

## Running the Core Benchmark

### 1. Train localization models

```bash
python -m scripts.train_baselines
```

This command trains the NN, Linear Regression, and XGBoost baselines and writes
the trained artifacts to `models/`.

### 2. Run exact enumeration

```bash
python -m scripts.run_bruteforce --backend nn  --kmin 1 --kmax 21
python -m scripts.run_bruteforce --backend xgb --kmin 1 --kmax 21
```

Brute force returns the exact maximum localization error for each evaluated
attack budget $k$. Runtime grows rapidly near the middle of the subset space.

### 3. Run scalable and baseline methods

```bash
python -m scripts.run_genetic_algorithm   --model-kind nn
python -m scripts.run_simulated_annealing --model-kind nn
python -m scripts.run_rssi_proximity      --model-kind nn
python -m scripts.run_random_sampling     --model-kind nn
```

Use `--model-kind xgb` to evaluate the corresponding XGBoost model. Each script
also accepts explicit training, validation, noise, model, seed, and output paths;
run a command with `--help` for its complete interface.

## Output

The scripts create:

```text
models/     # trained localization models and scalers
results/    # exact and approximate-search results
```

Each result records the attack budget, selected AP subset, localization error,
and method-specific runtime information required for subsequent analysis.

## Main Findings

- Exact enumeration identifies rare, high-impact AP subsets that typical
  perturbations do not reveal.
- The most damaging subset depends on the attack budget $k$, localization
  model, AP representation, and indoor environment.
- Strong average RSSI is not a reliable indicator of AP criticality.
- SA provides the closest overall approximation to exact enumeration, with GA
  following closely.
- GA and SA remain effective in expanded AP spaces where complete enumeration
  is computationally infeasible.
- Structured search reduces assessment time by orders of magnitude relative to
  exhaustive evaluation.

## Citation

If you use this repository, please cite:

```bibtex
@article{noshin2026combinatorial,
  author  = {Muhammed Noshin and Muhammad Hannan Akhtar and
             Mohamed I. AlHajri and Mohammad Zulkernine},
  title   = {On the Combinatorial Vulnerability of Access Points in
             Wi-Fi Fingerprinting: Characterization and Efficient Identification},
  year    = {2026},
  note    = {Manuscript under review}
}
```

The citation will be updated with the journal, volume, pages, and DOI after
publication.

## Authors and Affiliations

**Muhammed Noshin**, **Muhammad Hannan Akhtar**, **Mohamed I. AlHajri**, and
**Mohammad Zulkernine**

- Department of Computer Science and Engineering, American University of
  Sharjah, United Arab Emirates
- School of Computing, Queen's University, Canada

## License

This repository is released under the MIT License. Dataset licenses remain
governed by their original providers.
