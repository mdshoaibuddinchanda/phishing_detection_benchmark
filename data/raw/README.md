# Dataset Preparation Instructions

## Required Dataset Format

The pipeline expects a CSV file with the following columns:

- `text`: Email content (string)
- `label`: Binary classification (0 = legitimate, 1 = phishing)

## Recommended Datasets

### 1. Kaggle Phishing Email Dataset

**Recommended Dataset (Used in this project)*

```bash
# Download from: https://doi.org/10.34740/kaggle/ds/5074342
# Or via Kaggle: https://www.kaggle.com/datasets/phishing-email-dataset
# Place in: data/raw/phishing_emails.csv
```

**Citation:**

```bibtex
@misc{phishing_kaggle_5074342,
  title        = {Phishing Email Dataset},
  author       = {Kaggle Datasets},
  year         = {2024},
  doi          = {10.34740/kaggle/ds/5074342},
  url          = {https://doi.org/10.34740/kaggle/ds/5074342}
}
```

### 2. Enron Email Dataset (with phishing labels)

```bash
# Download from: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
# Requires manual labeling or use pre-labeled version
```

### 3. Custom Dataset

Create CSV with format:

```csv
text,label
"Dear customer, verify your account at http://phishing.com",1
"Meeting scheduled for tomorrow at 3 PM",0
```

## Data Placement

```bash
eco_phish/
└── data/
    └── raw/
        └── phishing_emails.csv  # Place your dataset here
```

## Dataset Size Recommendations

- **Minimum**: 5,000 samples
- **Recommended**: 10,000+ samples
- **Balanced classes**: Aim for ~50/50 split between phishing and legitimate

## Preprocessing Notes

The pipeline will automatically:

- Convert text to lowercase
- Remove URLs and email addresses
- Clean special characters
- Remove empty samples
- Split into train (70%), val (15%), test (15%)

**No manual preprocessing required.**
