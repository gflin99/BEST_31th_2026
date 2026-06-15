# BEST31: Models Framework for Fermentation Process Prediction

This repository contains the official implementation for our research paper. It features a rigorous evaluation framework for bio-process time-series prediction using Recurrent Neural Networks (RNN) and Autoencoders.

## Research Highlights
* **Rigorous Validation:** Implements **Leave-One-Bio-Group-Out Cross Validation (LOBOCV)** across N_exp independent fermentation experiments (`ExpID`) to guarantee out-of-batch generalization.
* **Robustness & Reproducibility:** Evaluated across **10 different random seeds** to ensure statistical significance.
* **Transfer Learning Architecture:** Features pre-trained **Autoencoders (AE)** integrated with **GRU/LSTM** networks, benchmarked against pure recurrent baselines.

---

## Data Privacy & Synthetic Dataset
To protect proprietary research data and maintain confidentiality prior to official publication, the source experimental datasets are **not publicly disclosed**. 

Instead, we provide a **synthetic dummy dataset** (`LAB_train2025_sample.csv`) in this repository. This dataset contains randomized variables with identical feature dimensions (`ExpID`, `HOUR`, `OD600`, etc.). It is provided **strictly for reviewers and scholars to verify the execution of the computational pipeline** and environment setup.

---

## Repository Structure
* `four_models_lobocv.py`: Main execution script containing model architectures, LOBOCV loop, and evaluation.
* `LAB_train2025_sample.csv`: Synthetic dummy dataset for pipeline verification.

---

## Environment Setup

Ensure you have Python 3.8+ installed. Install the required dependencies via `pip`:

```bash
pip install tensorflow numpy pandas scikit-learn
```

---

## How to Reproduce My Results

### 1. Quick Verification (Highly Recommended for Reviewers)
The full experiment (N_exp experiments × N_seed seeds × 4 models) involves **360 training runs in my study** and may take hours. To quickly verify the pipeline functionality on the sample data, run with reduced epochs:

```bash
python four_models_lobocv.py --data LAB_train2025_sample.csv --ae-epochs 5 --gru-epochs 5 --output quick_test_results.csv
```

### 2. Full Reproduction
To execute the complete benchmarking suite matching the paper's settings:

```bash
python four_models_lobocv.py --data LAB_train2025_sample.csv
```

---

## Expected Outputs
Upon completion, the script automatically exports `results_four_models.csv` and prints two summary tables to the terminal:
1. **Overall Performance:** Mean, standard deviation, min, and max of $R^2$ scores across all seeds and groups for each model type (`AE-GRU`, `AE-LSTM`, `GRU`, `LSTM`).
2. **Detailed Breakdown:** Cross-tabulated mean $R^2$ performance per `test_exp` block.
