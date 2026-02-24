# Data_Generation_using_Modeling_Simulation

> Synthetic dataset generation via discrete-event simulation (M/M/1 Queue) + binary classification benchmarking across 8 ML models.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![SimPy](https://img.shields.io/badge/SimPy-Discrete--Event%20Simulation-green)](https://simpy.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](https://jupyter.org)

---

## 📌 Project Overview

This project demonstrates how **discrete-event simulation** can generate rich synthetic datasets for Machine Learning — removing the need for expensive real-world data collection.

The pipeline has two stages:

1. **Simulate** — A SimPy-based M/M/1 queue model runs 1,000 experiments across randomized input parameters, recording the average customer wait time per run.
2. **Classify** — The continuous wait time is converted into a **binary class label** (High Wait / Low Wait), and 8 ML classifiers are trained and benchmarked against each other.

---

## 🧠 System Model: The M/M/1 Queue

The simulation models a **single-server queuing system** — one server (teller/ATM) handling a stream of randomly arriving customers, each with a randomly sampled service duration.

```
Customers Arrive → [Waiting Queue] → [Single Server] → Exit
(Exponential inter-arrivals)           (Exponential service times)
```

Both inter-arrival times and service times follow an **exponential distribution**, consistent with the classical M/M/1 model from queuing theory. Each run processes **120 customers**.

---

## ⚙️ Simulation Parameters

| Parameter | Description | Lower Bound | Upper Bound |
|---|---|---|---|
| `mean_interarrival` | Avg. minutes between customer arrivals | **1.5 min** | **9.5 min** |
| `mean_service` | Avg. minutes per customer transaction | **0.8 min** | **2.2 min** |
| `avg_wait` | Recorded avg. wait time → used to derive label | — | — |

Parameters are sampled uniformly at random for each of the 1,000 simulation runs (`np.random.uniform`). Seeds are fixed (`random.seed(42)`, `np.random.seed(42)`) for full reproducibility.

---


## 🤖 ML Models & Evaluation

The dataset is split **75% train / 25% test** (`test_size=0.25`, `random_state=42`).

Eight classifiers are trained and compared:

| # | Model | Key Config |
|---|---|---|
| 1 | **Random Forest** | `n_estimators=100` |
| 2 | **Gradient Boosting** | `n_estimators=100` |
| 3 | **AdaBoost** | `n_estimators=100` |
| 4 | **SVM (RBF Kernel)** | `kernel='rbf'`, `probability=True` |
| 5 | **KNN** | `k=5` |
| 6 | **Logistic Regression** | `max_iter=1000` |
| 7 | **Decision Tree** | Default depth |
| 8 | **Naive Bayes** | Gaussian |

Each model is evaluated on four metrics:

| Metric | What it measures |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted "High Wait", how many were actually high |
| **Recall** | Of actual "High Wait" cases, how many were caught |
| **F1 Score** | Harmonic mean of Precision & Recall |

Results are printed sorted by **Accuracy (descending)**, and the best-performing model is highlighted automatically.

---

## 🚀 Getting Started

### ▶ Run in Google Colab (No setup needed)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xeMrx7l7QRyFB4yvii2E0Pl2PZszYnIF#scrollTo=lfdCx8BOwbwk)

The notebook installs SimPy automatically with `!pip install simpy`.

### 💻 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Kumarkashish511/Data_Generation_using_Modeling_Simulation.git
cd Data_Generation_using_Modeling_Simulation
```

**2. Install dependencies**
```bash
pip install simpy numpy pandas scikit-learn
```

**3. Launch the notebook**
```bash
jupyter notebook Data_Generation.ipynb
```

---

## 📁 Repository Structure

```
Data_Generation_using_Modeling_Simulation/
│
├── Data_Generation.ipynb    # Full pipeline: simulation → labeling → ML training & evaluation
└── README.md                # Project documentation
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `simpy` | Discrete-event simulation engine |
| `numpy` | Random sampling & numerical ops |
| `pandas` | Dataset construction & management |
| `scikit-learn` | ML models, train/test split, metrics |

---

## 💡 Why This Approach?

**Why use simulation to generate data?**
Real-world queuing data is slow and expensive to collect. A simulator lets you instantly sample any combination of arrival and service rates — including extreme edge cases — with zero collection cost and perfectly clean labels.

**Why convert wait time to a binary class?**
Framing the problem as **classification** (High vs. Low congestion) maps directly to real-world decisions: *"Should we open another counter?"* Binary labels also make model comparison straightforward using standard classification metrics.

**Why is the dataset balanced by design?**
Using the **median** as the class threshold ensures exactly half the samples fall in each class, so no model gets a free accuracy boost from class imbalance.

---
## Result

<img width="758" height="399" alt="image" src="https://github.com/user-attachments/assets/55af0abe-3bc1-44da-ba2e-337964d6e330" />


## 🔁 Reproducing Results

The random seeds are fixed throughout:
```python
random.seed(42)
np.random.seed(42)
# train_test_split(..., random_state=42)
```
Running the notebook end-to-end will produce identical results every time.
