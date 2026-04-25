# On the Combinatorial Vulnerability of Access Points in Wi-Fi Fingerprinting

![Figure 1: Methodology for combinatorial AP vulnerability assessment](figures/figure1.pdf)

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

To enable scalable vulnerability assessment in practical IoT deployments, we evaluate a range of search strategies for identifying high-impact AP combinations. The results show that structured combinatorial search can efficiently recover near-worst-case attacks without exhaustive enumeration. These findings establish combinatorial vulnerability as a fundamental property of Wi-Fi fingerprinting systems and highlight the need for systematic, infrastructure-level security evaluation.

---

## Setup

### 1. Clone and prepare environment

```bash
git clone https://github.com/HannanAkhtar/APexDQN-Deep-Q-Network-based-Access-Point-Attack-Optimization-for-Indoor-Localization-Security.git
cd APexDQN-Deep-Q-Network-based-Access-Point-Attack-Optimization-for-Indoor-Localization-Security
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
python -m scripts.run_search --method ga --backend xgb
python -m scripts.run_search --method sa --backend xgb
python -m scripts.run_search --method dqn --backend xgb
python -m scripts.run_search --method rssi --backend xgb
python -m scripts.run_search --method random --backend xgb
```

Each method evaluates candidate AP subsets under the same perturbation protocol and reports the resulting attacked-set localization error.

---

## Methodology Summary

### Brute-Force Enumeration

Brute-force search exhaustively evaluates every possible AP subset for a fixed attack size *k*. This provides the true worst-case AP combination and serves as the reference for evaluating all approximate search methods.

### Genetic Algorithm (GA)

GA searches directly over fixed-size AP subsets using population-based evolution. Candidate subsets are evaluated according to the localization degradation they induce, and crossover, mutation, and elitism are used to explore the combinatorial search space.

### Simulated Annealing (SA)

SA searches over fixed-size AP subsets by repeatedly proposing single-swap neighbors. It accepts improving moves and occasionally accepts worse moves according to a cooling schedule, allowing it to escape poor local optima. In the experiments, SA provides the strongest runtime-accuracy trade-off.

### Deep Q-Network (DQN)

The DQN-based method is formulated as a one-step contextual bandit. Each action corresponds to a complete AP subset, and the immediate reward is the vulnerability score induced by perturbing that subset. In this version of the paper, DQN is treated as one benchmarked search strategy rather than the central contribution.

### RSSI-Proximity and Random Sampling

RSSI-proximity tests whether stronger average signal strength is a reliable proxy for AP criticality, while random sampling provides an uninformed baseline. Both methods are computationally cheap but consistently underestimate worst-case vulnerability.

---

## Experimental Results

| Result Category | Main Finding |
|:---|:---|
| Baseline performance | XGBoost achieves the lowest baseline localization error, while NN is also competitive. |
| Brute-force vulnerability | NN reaches a worst-case RMSE of about 24.3 m at k = 14, while XGBoost plateaus near 18.8 m. |
| Vulnerability distribution | Worst-case AP subsets appear as rare upper-tail outliers rather than typical attack cases. |
| RSSI-proximity heuristic | Strong AP signal strength does not reliably identify the most damaging AP subsets. |
| Search methods | GA and SA most consistently recover near-brute-force AP subsets. |
| Runtime | SA provides the best runtime-accuracy trade-off, achieving large speedups over brute-force while remaining close to the worst-case upper bound. |

Overall, the results show that effective localization security auditing requires structured combinatorial search rather than average-case testing, random sampling, or signal-strength-based AP ranking.

---

## Key Contributions

- Formulates **combinatorial AP vulnerability assessment** for Wi-Fi fingerprint-based indoor localization.
- Characterizes the vulnerability landscape using exhaustive enumeration on the UJIIndoorLoc dataset.
- Shows that worst-case AP subsets are rare, high-impact combinations in the upper tail of the attack distribution.
- Demonstrates that RSSI strength alone is an unreliable indicator of AP criticality.
- Benchmarks brute-force, heuristic, metaheuristic, and learning-based search strategies on NN and XGBoost localization models.
- Shows that GA and SA can recover near-worst-case AP subsets without exhaustive enumeration, with SA offering the strongest runtime-accuracy trade-off.

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