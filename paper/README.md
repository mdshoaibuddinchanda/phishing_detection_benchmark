# Paper Writing Guide

## Figures to Include

All figures are automatically generated in `results/figures/`. Copy needed figures to `paper/figures/` for LaTeX compilation.

### Required Figures

1. **pareto_frontier.png** - Main result showing accuracy-energy trade-off
2. **accuracy_comparison.png** - Performance metrics comparison
3. **energy_comparison.png** - Energy consumption comparison

### Optional Figures

- all_metrics.png
- co2_comparison.png
- latency_comparison.png
- model_size_comparison.png
- multi_objective.png

## Table Guidelines

The main results table should include:

| Column       | Description               |
|--------------|---------------------------|
| Model        | Model name                |
| Accuracy     | Classification accuracy   |
| Precision    | Precision score           |
| Recall       | Recall score              |
| F1-Score     | F1-score (primary metric) |
| Size (MB)    | Model size                |
| Latency (ms) | Per-sample inference time |
| Energy (kWh) | Total energy consumption  |
| CO₂ (g)      | CO₂ emissions             |

Export from: `results/tables/results_summary.csv`

## LaTeX Compilation

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## IEEE Template

Download official template from conference website and replace `paper.tex` header with conference-specific formatting.

## Submission Checklist

- [ ] All figures in `paper/figures/`
- [ ] Results table populated with actual values
- [ ] Abstract complete (250 words max)
- [ ] References formatted correctly
- [ ] Author information updated
- [ ] Figures referenced in text
- [ ] Tables referenced in text
- [ ] Acknowledgments section (if needed)
- [ ] Proofread for typos
- [ ] Check page limit (typically 6-8 pages for IEEE)
