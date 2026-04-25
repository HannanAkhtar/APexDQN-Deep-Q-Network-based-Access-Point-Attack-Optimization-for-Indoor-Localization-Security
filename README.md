# On the Combinatorial Vulnerability of Access Points in Wi-Fi Fingerprinting: Characterization and Efficient Identification

![Figure 1: Methodology for combinatorial AP vulnerability assessment](figures/figure1.pNG)

---

## Overview

This repository provides the official implementation of the paper:

> **Muhammed Noshin**, **Muhammad Hannan Akhtar**, **Mohamed I. AlHajri**, **Mohammad Zulkernine**  
> *On the Combinatorial Vulnerability of Access Points in Wi-Fi Fingerprinting: Characterization and Efficient Identification*, 2026.

This work studies the security of Wi-Fi fingerprint-based indoor localization systems under adversarial manipulation of access point (AP) signals. We formulate the problem as a **combinatorial AP vulnerability assessment** task: given a set of deployed APs, the goal is to identify which subset of APs causes the largest degradation in localization accuracy when perturbed.

Using the UJIIndoorLoc dataset, we exhaustively characterize the vulnerability landscape over retained AP subsets and show that worst-case localization failures are highly non-uniform and heavy-tailed. Most AP combinations cause moderate degradation, while a small number of rare AP subsets lead to disproportionately large localization errors.

The repository benchmarks brute-force enumeration against scalable search strategies, including Genetic Algorithm (GA), Simulated Annealing (SA), Deep Q-Network (DQN), RSSI-proximity selection, and random sampling. The results show that GA and SA most reliably recover near-worst-case AP subsets, while RSSI-based and random strategies substantially underestimate worst-case vulnerability.

---

## Abstract

Wi-Fi fingerprint-based indoor localization is widely deployed in Internet of Things (IoT)-enabled environments, where accurate positioning is critical for applications such as navigation, asset tracking, and automation. However, the security of these systems under adversarial manipulation of access points remains insufficiently understood.

In particular, a fundamental question arises: **which subsets of APs, if compromised, induce the worst-case degradation in localization performance?** This paper formulates the problem as a combinatorial AP vulnerability assessment task and provides a systematic characterization of its structure.

Through exhaustive analysis on the UJIIndoorLoc dataset, we show that localization vulnerability is highly non-uniform and exhibits a heavy-tailed distribution: most AP subsets produce only moderate degradation, while a small number of rare combinations cause disproportionately large localization errors. We further demonstrate that common defensive intuitions, such as prioritizing high-RSSI APs, fail to identify these security-critical subsets.

To enable scalable vulnerability assessment in practical IoT deployments, we evaluate a range of search strategies for identifying high-impact AP combinations. The results show that structured combinatorial search can efficiently recover near-worst-case attacks without exhaustive enumeration. In particular, GA and SA recover near-brute-force AP subsets, with SA achieving speedups exceeding **5000× on NN** and up to approximately **3000× on XGBoost** at peak brute-force complexity. These findings establish combinatorial vulnerability as a fundamental property of Wi-Fi fingerprinting systems and highlight the need for systematic, infrastructure-level security evaluation.

---

## Setup

### 1. Clone and prepare environment

```bash
git clone https://github.com/HannanAkhtar/AP-Vulnerability-WiFi-Localization.git
cd AP-Vulnerability-WiFi-Localization
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dependencies

```text
numpy, pandas, scikit-learn, tensorflow, keras, torch, xgboost, tqdm, matplotlib, joblib
```

The `data/` folder contains the CSV files and configuration files used in the experiments.

---

## Running Experiments

### 1. Train Baseline Models

Train and evaluate the baseline localization models:

```bash
python -m scripts.train_baselines
```

The baseline models are trained on the filtered UJIIndoorLoc RSSI features and evaluated using localization error in meters.

---

### 2. Brute-Force Vulnerability Analysis

Enumerate all AP subsets of size *k = 1, ..., 21* and measure the induced localization degradation:

```bash
python -m scripts.run_bruteforce --backend xgb
```

Brute-force enumeration provides the ground-truth upper bound for each attack size and is used to characterize the vulnerability landscape.

---

### 3. Search-Based Vulnerability Identification

Run scalable search methods to identify high-impact AP subsets without exhaustive enumeration:

```bash
python -m scripts.run_genetic_algorithm --backend xgb
python -m scripts.run_simulated_annealing --backend xgb
python -m scripts.train_dqn --backend xgb
python -m scripts.run_rssi_proximity --backend xgb
python -m scripts.run_random_sampling --backend xgb
```

Each method evaluates candidate AP subsets under the same perturbation protocol and reports the resulting attacked-set localization error.

---

## Methodology Summary

### Brute-Force Enumeration

Brute-force search exhaustively evaluates every possible AP subset for a fixed attack size *k*. This provides the true worst-case AP combination and serves as the reference for evaluating all approximate search methods.

For the retained 21-AP setting, the total search space contains:

```text
2^21 - 1 = 2,097,151 non-empty AP subsets
```

This exhaustive search is feasible for the filtered 21-AP setting, but the runtime grows sharply near the middle of the search space, making brute-force impractical for routine assessment in larger deployments.

### Genetic Algorithm (GA)

GA searches directly over fixed-size AP subsets using population-based evolution. Candidate subsets are evaluated according to the localization degradation they induce, and crossover, mutation, and elitism are used to explore the combinatorial search space.

GA consistently recovers near-worst-case AP subsets and closely tracks the brute-force upper bound across both NN and XGBoost models.

### Simulated Annealing (SA)

SA searches over fixed-size AP subsets by repeatedly proposing single-swap neighbors. It accepts improving moves and occasionally accepts worse moves according to a cooling schedule, allowing it to escape poor local optima.

SA provides the strongest runtime-accuracy trade-off in the experiments. It remains close to the brute-force upper bound while reducing the runtime from hours or minutes to seconds or sub-seconds, depending on the model.

### Deep Q-Network (DQN)

The DQN-based method is formulated as a one-step contextual bandit. Each action corresponds to a complete AP subset, and the immediate reward is the vulnerability score induced by perturbing that subset.

In this version of the paper, DQN is treated as one benchmarked search strategy rather than the central contribution. DQN captures the general vulnerability trend but remains weaker than GA and SA in recovering the most damaging upper-tail AP subsets.

### RSSI-Proximity and Random Sampling

RSSI-proximity tests whether stronger average signal strength is a reliable proxy for AP criticality, while random sampling provides an uninformed baseline.

Both methods are computationally cheap but consistently underestimate worst-case vulnerability. This shows that signal strength alone is not a reliable indicator of AP criticality and that the most damaging AP subsets arise from higher-order AP interactions.

---

## Experimental Results

### Baseline Localization Performance

| Model | RMSE (m) |
|:---|---:|
| Linear Regression | 20.86 |
| XGBoost | 10.31 |
| LightGBM | 11.98 |
| Neural Network | 11.30 |

XGBoost achieves the lowest baseline localization error, while the Neural Network model is also competitive. These two models are therefore used for the main vulnerability analysis.

---

### Brute-Force Vulnerability Analysis

| Model | Baseline RMSE | Worst-Case RMSE | Attack Size | Increase |
|:---|---:|---:|---:|---:|
| Neural Network | 11.30 m | ≈ 24.3 m | k = 14 | ≈ 115% |
| XGBoost | 10.31 m | ≈ 18.8 m | plateau region | ≈ 82% |

The Neural Network model is more sharply affected by AP perturbation, reaching a worst-case RMSE of about **24.3 m** at **k = 14**. XGBoost shows a smoother and more resilient degradation profile, plateauing near **18.8 m RMSE**.

---

### Vulnerability Landscape

The brute-force results show that the vulnerability landscape is highly non-uniform and heavy-tailed. Most AP subsets produce moderate degradation, while a small number of rare AP combinations produce extreme localization errors.

This means that worst-case localization failures are not representative of typical AP perturbations. Random failures or uninformed AP subset selection can therefore significantly underestimate the true worst-case risk.

---

### Search Method Quality

| Method | Main Observation |
|:---|:---|
| Brute Force | Provides the ground-truth worst-case AP subsets. |
| GA | Closely tracks brute-force and reliably recovers near-worst-case subsets. |
| SA | Closely tracks brute-force and provides the best runtime-accuracy trade-off. |
| DQN | Captures the general trend but remains below GA and SA in the upper-tail region. |
| RSSI-Proximity | Fails to identify the most damaging AP subsets despite using signal-strength information. |
| Random Sampling | Computationally cheap but substantially underestimates worst-case vulnerability. |

For the NN model, GA and SA remain especially close to brute force in the high-risk region. For example:

| Attack Size | Brute Force RMSE | GA / SA RMSE |
|:---|---:|---:|
| k = 5 | ≈ 16.77 m | ≈ 16.77 m |
| k = 11 | ≈ 23.91 m | ≈ 23.91 m |
| k = 14 | ≈ 24.34 m | ≈ 24.33 m |

DQN remains weaker in the same region. For example, on the NN model, it reaches about **14.89 m** at k = 5, **19.18 m** at k = 11, and **20.85 m** at k = 14.

At k = 14, RSSI-proximity and random sampling reach only about **15.90 m** and **15.69 m**, respectively, which is far below the brute-force worst case.

For XGBoost, GA and SA also closely track brute force. For example:

| Attack Size | GA / SA RMSE |
|:---|---:|
| k = 10 | ≈ 18.62 m |
| k = 12 | ≈ 18.71 m |
| k = 17 | ≈ 18.78 m |

DQN performs better on XGBoost than on NN but still remains below the strongest upper-tail results. At k = 14, RSSI-proximity and random sampling reach only about **16.12 m** and **15.65 m**, respectively.

---

### Runtime Analysis

| Model | Brute-Force Peak Runtime | SA Runtime | SA Speedup |
|:---|---:|---:|---:|
| Neural Network | ≈ 268 minutes near k = 10–11 | ≈ 2.83–3.03 seconds | > 5000× |
| XGBoost | ≈ 15–17 minutes near peak complexity | ≈ 0.31–0.51 seconds | up to ≈ 3000× |

The runtime analysis shows that brute-force search becomes expensive near the middle of the AP subset space, where the number of candidate combinations is largest.

SA reduces the NN brute-force search from several hours to only a few seconds, achieving speedups exceeding **5000×**. For XGBoost, SA reduces the peak brute-force runtime from around 15–17 minutes to sub-second execution, achieving speedups up to approximately **3000×**.

Overall, SA offers the strongest practical balance: it remains close to the brute-force upper bound while requiring only a fraction of the runtime.

---

## Key Contributions

- Formulates **combinatorial AP vulnerability assessment** for Wi-Fi fingerprint-based indoor localization.
- Characterizes the vulnerability landscape using exhaustive enumeration on the UJIIndoorLoc dataset.
- Shows that worst-case AP subsets are rare, high-impact combinations in the upper tail of the attack distribution.
- Demonstrates that RSSI strength alone is an unreliable indicator of AP criticality.
- Benchmarks brute-force, heuristic, metaheuristic, and learning-based search strategies on NN and XGBoost localization models.
- Shows that GA and SA recover near-worst-case AP subsets without exhaustive enumeration.
- Demonstrates that SA provides the strongest runtime-accuracy trade-off, achieving speedups exceeding **5000× on NN** and up to approximately **3000× on XGBoost** at peak brute-force complexity.

---

## Citation

If you use this code, please cite:

```bibtex
@article{noshin2026combinatorial,
  author  = {Muhammed Noshin and Muhammad Hannan Akhtar and Mohamed I. AlHajri and Mohammad Zulkernine},
  title   = {On the Combinatorial Vulnerability of Access Points in Wi-Fi Fingerprinting: Characterization and Efficient Identification},
  year    = {2026},
  note    = {Official implementation}
}
```

---

## Authors and Affiliations

**Muhammed Noshin**, **Muhammad Hannan Akhtar**, **Mohamed I. AlHajri**, **Mohammad Zulkernine**  
- *Department of Computer Science and Engineering*, American University of Sharjah, UAE  
- *School of Computing*, Queen's University, Canada

---

## License

Released under the **MIT License**.  
You are free to use, modify, and distribute this code with proper attribution.

---
