# Phishing Detection Benchmarking: Accuracy–Energy Trade-offs

**Research project for benchmarking Small Language Models on phishing detection with energy consumption analysis.**

## 📋 Overview

This repository contains a reproducible pipeline for evaluating the accuracy–energy trade-offs of transformer-based language models for phishing email detection. The project compares:

- **RoBERTa-Large** (heavyweight baseline)
- **DistilBERT** (lightweight transformer)
- **Phi-3-mini** (small language model)

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

```flie
eco_phish/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Train/val/test splits
├── models/                     # Saved model checkpoints
│   ├── bert_large/
│   ├── distilbert/
│   └── phi3_mini/
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

- GPU: [Specify your GPU, e.g., NVIDIA RTX 4090]
- CPU: [Specify your CPU]
- RAM: [Specify RAM]
- OS: Windows/Linux

## 📈 Typical Results

| Model         | Accuracy | F1-Score | Energy (kWh) | CO₂ (g)  | Latency (ms) | Size (MB) |
|---------------|----------|----------|--------------|----------|--------------|-----------|
| RoBERTa-Large | ~0.95    | ~0.94    | Higher       | Higher   | Higher       | ~1200     |
| DistilBERT    | ~0.93    | ~0.92    | Medium       | Medium   | Medium       | ~260      |
| Phi-3-mini    | ~0.92    | ~0.91    | **Lower**    | **Lower**| **Lower**    | ~2500     |

*Values are illustrative. Actual results depend on dataset and hardware.*

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{yourname2025phishing,
  title={Accuracy–Energy Trade-offs of Small Language Models for Real-Time Phishing Detection},
  author={Your Name},
  journal={IEEE Conference on AI},
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

## 🤝 Contributing

This is a research project. For questions or suggestions, please open an issue.

## ✅ Checklist for Submission

- [ ] Dataset placed in `data/raw/`
- [ ] Configuration verified in `config.yaml`
- [ ] All models trained successfully
- [ ] Benchmarking completed with energy logs
- [ ] Figures generated in `results/figures/`
- [ ] Results table saved in `results/tables/`
- [ ] Hardware details documented in README
- [ ] Paper draft references correct figure files
- [ ] Code tested and reproduces results

---

**Last updated:** December 2025  
**Contact:** [Your email]
