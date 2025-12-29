# Phishing Detection Benchmark Summary

## Title

- Transformer Models for Phishing Email Detection: Accuracy–Energy Trade-offs

## Contributions

- Config-driven, reproducible benchmarking pipeline for phishing email classification.
- Models compared: RoBERTa-large, RoBERTa-base, DistilBERT.
- Metrics: accuracy, precision, recall, F1, latency (ms/sample), energy (kWh), CO₂ (grams), model size (MB).
- Energy/CO₂ tracked with CodeCarbon; logs stored under results/logs.
- Publication-ready figures (accuracy, latency, energy/CO₂, model size, Pareto frontier) and results table (results/tables/results_summary.csv).

## Data & Preprocessing

- Source: data/raw/phishing_emails.csv with columns text, label (0=legit, 1=phish).
- Splits: stratified Train/Val/Test = 70/15/15, seed 42, saved under data/processed/.
- Cleaning: drop empty/duplicate/near-duplicate rows; whitespace normalization; discard very short messages before splitting.
- Tokenization after splitting; max length 256; truncate tail, pad per batch; no tokenizer fitting on full corpus.
- Leakage controls: hash-based split disjointness; deterministic preprocessing before split; fixed seed.

## Training Setup

- Epochs: 3; batch size 32 with grad accumulation 2 (effective 64).
- LR 2e-5, weight decay 0.01, warmup 500; AdamW, linear scheduler; grad clip 1.0; fp16 on GPU.
- Checkpoint each epoch (keep 3); resume by default; disable via --no-resume; force fresh run via --force-train.
- Max seq len 256; seed 42; workers 4; pin_memory true; optional gradient checkpointing.

## Inference & Evaluation

- Batched inference with torch.no_grad(); consistent batch sizes across models.
- Latency: wall-clock over full test set / sample count (includes warmup).
- Model size from on-disk checkpoints.
- Energy/CO₂: CodeCarbon process mode, offline, regional code configurable (default USA).
- Outputs: metrics table at results/tables/results_summary.csv; figures under results/figures/.

## Findings (qualitative)

- Accuracy: RoBERTa-large ≳ RoBERTa-base > DistilBERT (small gap).
- Latency/Energy: DistilBERT fastest/leanest; RoBERTa-base middle; RoBERTa-large slowest/most energy.
- Pareto: RoBERTa-base near frontier; DistilBERT anchors efficiency; RoBERTa-large anchors accuracy.
- Error patterns: FNs on benign-looking transactional mails with malicious links; FPs on urgent but legitimate notifications.

## Limitations

- Single text-only dataset; no headers/URL reputation/HTML.
- Energy estimates are approximate (process-level, hardware TDP, regional baseline).
- No adversarial robustness evaluation; moderate hyperparameters; results at seq len 256.
