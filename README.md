# Phishing Detection Benchmarking: Accuracy–Energy Trade-offs

**Research project for benchmarking transformer models on phishing email detection with energy consumption and latency analysis.**

## 📋 Overview

This repository contains a reproducible pipeline for evaluating the accuracy–energy trade-offs of transformer-based language models for phishing email detection. The project compares:

- **RoBERTa-Large** (heavyweight baseline)
- **RoBERTa-Base** (balanced baseline)
- **DistilBERT** (lightweight transformer)

### Key Metrics

**Performance:**

- Accuracy
- Precision
- Recall
- F1-Score

**Efficiency:**

- Inference latency (ms per sample)
- Model size (MB)
- Energy consumption (kWh)
- CO₂ emissions (grams)

## 🏗️ Project Structure

```text
eco_phish/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Train/val/test splits
├── models/                     # Saved model checkpoints
│   ├── roberta_large/
│   ├── roberta_base/
│   └── distilbert/
├── src/
│   ├── config/                 # Configuration management
│   ├── data/                   # Data loading and preprocessing
│   ├── training/               # Model training
│   ├── energy/                 # Energy tracking (CodeCarbon)
│   ├── inference/              # Benchmarking
│   ├── metrics/                # Performance metrics
│   ├── visualization/          # Plotting
│   └── main.py                 # Pipeline orchestrator
├── results/
│   ├── tables/                 # CSV results
│   ├── figures/                # Publication-ready plots
│   └── logs/                   # Energy logs
└── paper/                      # IEEE paper LaTeX files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- `uv` package manager

### Installation

```bash
# Clone repository
cd eco_phish

# Create virtual environment with uv
uv venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Dataset Setup

1. Download a phishing email dataset (e.g., from Kaggle)
2. Place CSV file in `data/raw/phishing_emails.csv`
3. Ensure columns: `text` (email content) and `label` (0/1 for legitimate/phishing)

### Running the Pipeline

**Complete pipeline (data → training → benchmarking → visualization):**

```bash
uv run python src/main.py
```

**Skip data preparation (use existing splits):**

```bash
uv run python src/main.py --skip-data
```

**Skip training (use existing models):**

```bash
uv run python src/main.py --skip-training
```

**Only generate visualizations from existing results:**

```bash
uv run python src/main.py --only-visualize
```

**Force retraining (even if trained artifacts exist):**

```bash
uv run python src/main.py --skip-data --force-train
```

**Retrain without resuming from checkpoints (fresh start):**

```bash
uv run python src/main.py --skip-data --force-train --no-resume
```

## 📊 Expected Outputs

### Tables

- `results/tables/results_summary.csv` - Complete metrics table for paper

### Figures

- `accuracy_comparison.png` - Accuracy vs F1-Score
- `all_metrics.png` - Complete performance metrics
- `energy_comparison.png` - Energy consumption
- `co2_comparison.png` - CO₂ emissions
- `latency_comparison.png` - Inference latency
- `model_size_comparison.png` - Model sizes
- `pareto_frontier.png` - Accuracy-energy trade-off
- `multi_objective.png` - Multi-objective comparison

## ⚙️ Configuration

All experimental parameters are centralized in `src/config/config.yaml`:

- Dataset paths and split ratios
- Model selections and hyperparameters
- Training configuration (epochs, batch size, learning rate)
- Energy tracking settings
- Visualization preferences

**To modify experiments, edit this file only.**

## 🔬 Reproducibility

This pipeline ensures reproducibility through:

1. **Fixed random seed** (42) across all operations
2. **Deterministic data splitting** with stratification
3. **Identical training hyperparameters** for all models
4. **Versioned dependencies** in `requirements.txt`
5. **Configuration-driven execution** (no hardcoded values)

### Hardware Used

- GPU: [NVIDIA RTX 3050]
- CPU: [intel i5-11300H]
- RAM: [32 GB]
- OS: Windows 11

## 📈 Typical Results

| Model           | Accuracy | F1-Score | Energy (kWh) | CO₂ (g) | Latency (ms) | Size (MB) |
|-----------------|----------|----------|--------------|---------|--------------|-----------|
| DistilBERT      | 0.9932   | 0.9935   | 0.0031       | 3.14    | 9.6          | 2,558     |
| ALBERT-Base     | 0.9923   | 0.9926   | 0.0067       | 6.72    | 20.9         | 455       |
| BERT-Base       | 0.9933   | 0.9935   | 0.0059       | 5.95    | 18.8         | 4,181     |
| RoBERTa-Base    | 0.9939   | 0.9942   | 0.0056       | 5.58    | 17.4         | 4,774     |
| ELECTRA-Base    | 0.9938   | 0.9940   | 0.0060       | 6.05    | 18.8         | 4,181     |
| DeBERTa-v3-Base | 0.9939   | 0.9941   | 0.0079       | 7.92    | 24.9         | 7,078     |
| ALBERT-Large    | 0.9950   | 0.9952   | 0.0215       | 21.50   | 68.0         | 686       |
| BERT-Large      | 0.9954   | 0.9956   | 0.0193       | 19.35   | 61.1         | 12,790    |
| RoBERTa-Large   | 0.9951   | 0.9953   | 0.0183       | 18.31   | 58.1         | 13,570    |

*Results from benchmarking on held-out test set (12,375 samples).*

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{chanda2025phishing,
  title={Accuracy–Energy Trade-offs of Transformer Models for Phishing Email Detection},
  author={ MD Shoaib Uddin Chanda},
  journal={IEEE/ACM Transactions on Machine Learning and Systems},
  year={2025}
}
```

## 🛠️ Troubleshooting

**Out of memory during training:**

- Reduce `batch_size` in `config.yaml`
- Use gradient accumulation
- Switch to smaller max_length (e.g., 256)

**Energy tracking not working:**

- Ensure CodeCarbon is installed: `uv pip install codecarbon`
- Check country ISO code in config
- Verify write permissions for logs directory

**Model download fails:**

- Check internet connection
- Models will auto-download from HuggingFace on first run
- Alternatively, pre-download models

## 📄 License

MIT License
