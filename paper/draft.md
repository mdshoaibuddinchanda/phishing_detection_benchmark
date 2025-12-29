# Transformer Models for Phishing Email Detection: Accuracy–Energy Trade-offs

## Abstract (150–200 words)

Phishing emails remain a leading vector for credential theft and fraud. We present a reproducible, config-driven benchmarking pipeline that evaluates transformer models for phishing email classification across accuracy, latency, energy consumption, and estimated CO₂ emissions. Using a labeled email corpus, we compare RoBERTa-large, RoBERTa-base, and DistilBERT under identical data splits, preprocessing, and training hyperparameters. Energy and emissions are measured with CodeCarbon to capture environmental impact. The pipeline produces publication-ready tables and figures, including Pareto frontiers highlighting accuracy–efficiency trade-offs. Results show that larger models offer slight accuracy gains at substantially higher latency and energy cost, while RoBERTa-base provides a practical balance and DistilBERT excels in speed and efficiency. The configuration-only workflow minimizes code changes, reduces leakage risk, and enables transparent, defensible reporting for production-oriented deployments.

## Introduction (900–1,200 words)

Phishing continues to be one of the most pervasive and costly cyber threats, leveraging social engineering and carefully crafted emails to steal credentials, distribute malware, and authorize fraudulent transactions. Email remains the dominant channel because it is universal, inexpensive for attackers, and familiar to users, making detection a persistent challenge. Traditional rule-based filters and keyword heuristics struggle against evolving attacker tactics, such as brand spoofing, URL obfuscation, and context-aware lures. As phishing kits and generative tools proliferate, defenders require detection methods that generalize beyond brittle indicators and capture nuanced semantic cues.

Machine learning has transformed phishing detection by moving beyond handcrafted signals toward models that learn patterns from data. Early ML approaches relied on classical features—URL tokens, sender domains, bag-of-words representations—and linear or tree-based classifiers. While lightweight, these systems are brittle under content mutation and often miss subtle intent. Deep neural models, particularly transformers pretrained on large corpora (e.g., BERT, RoBERTa), offer richer language understanding and have become state of the art for many security NLP tasks. Fine-tuning these models on labeled phishing corpora yields strong accuracy, capturing relationships between call-to-action phrases, brand impersonation, and linguistic anomalies that elude simpler models.

However, most evaluations remain narrowly accuracy-centric. Production email gateways and SOC workflows must also satisfy latency, throughput, and sustainability constraints. High-throughput filters need per-message inference measured in milliseconds, not seconds. Organizations increasingly track energy consumption and CO₂ emissions for sustainability reporting and cost control. A model that is marginally more accurate but significantly slower or more energy-intensive may be unsuitable for deployment. Despite this, public comparisons rarely report runtime, energy, or emissions, and cross-model benchmarking under a unified pipeline is uncommon.

The resulting gap is twofold. First, there is limited systematic evaluation of multiple transformer models for phishing detection that jointly considers accuracy, latency, energy, and emissions. Second, many pipelines are ad hoc and code-heavy, making reproducibility and leakage control difficult. A config-driven approach that centralizes all experimental settings (data paths, splits, models, hyperparameters, energy tracking) can reduce accidental changes, support re-runs, and simplify peer review.

This work addresses these gaps by releasing and applying a reproducible benchmarking pipeline focused on phishing email classification. The pipeline harmonizes data preparation, training, benchmarking, and visualization under a single YAML configuration, avoiding hardcoded hyperparameters. It compares three representative transformer variants—RoBERTa-large (high capacity), RoBERTa-base (balanced), and DistilBERT (efficient)—to map accuracy–efficiency trade-offs. Energy and emissions are measured with CodeCarbon to provide transparent environmental accounting. All outputs (metrics table, energy/latency plots, Pareto frontiers) are generated automatically for publication-ready reporting.

Scope and positioning: we concentrate on supervised binary classification of phishing vs legitimate emails using pretrained Hugging Face models fine-tuned on a labeled phishing corpus. We do not claim architectural novelty; the emphasis is on fair, reproducible comparison and inclusion of efficiency metrics that are often omitted. We also avoid overclaiming reproducibility: while seeds are fixed and splits are deterministic, GPU nondeterminism and library versions can affect bitwise repeatability. Nonetheless, the configuration-only design, fixed random seed, and stratified splits provide a solid baseline for defensible evaluation.

Operational perspective: enterprise email gateways process millions of messages per day, leaving little room for heavy per-message compute. Latency budgets are often single-digit milliseconds, and model footprint influences both horizontal scaling and hardware spend. SOC analysts need clear, auditable evidence for why a model is chosen, and sustainability teams increasingly request emissions numbers for always-on services. By surfacing energy/CO₂ metrics next to accuracy, the pipeline aligns technical evaluation with operational and environmental objectives.

Threat evolution: attackers continually adopt template mutation, homograph domains, benign-looking pretext text, and now large language models to craft fluent, personalized lures. A detector must remain robust to distribution shift, which motivates evaluating both higher-capacity models (for generalization) and efficient models (for real-time defense and rapid rollout). The benchmark does not attempt to exhaust adversarial tactics but aims to provide a defensible, repeatable baseline that others can extend with robustness tests.

Reproducibility stance: all tunable elements—data locations, tokenizer names, model checkpoints, training and inference batch sizes, resume/force-train flags—live in a single config file. This minimizes configuration drift and enables reruns with the exact same settings. Automated figures and tables reduce manual post-processing, lowering the risk of selection bias. We report defaults that prioritize stability and fairness across models rather than aggressive hyperparameter tuning for a single variant.

Threat model and deployment contexts: We target a passive inspection pipeline that classifies emails before delivery or flags them for SOC review. We do not modify messages or interact with senders. The model observes only message text (body and subject) and does not rely on network-layer signals, making it suitable for offline processing of historical archives or privacy-constrained environments. For inline use, inference latency is the primary constraint; for batch triage of historical mailboxes, throughput and energy dominate. Our evaluation emphasizes both settings by reporting per-sample latency and full-pass runtime.

Paper structure: Section 2 surveys related work across phishing detection, transformers in security, and efficiency reporting. Section 3 details the dataset, preprocessing, and leakage controls. Section 4 describes models, training, evaluation metrics, and energy tracking. Section 5 reports results and trade-off visualizations. Section 6 discusses operational guidance and sustainability implications. Section 7 covers limitations and Section 8 concludes.

Contributions of this work:

1) A reproducible, config-driven pipeline for phishing detection benchmarking that minimizes code changes and centralizes experimental control.
2) A multi-metric comparison of RoBERTa-large, RoBERTa-base, and DistilBERT covering accuracy, precision, recall, F1, latency, energy, CO₂, and model size.
3) Integrated energy and emission tracking via CodeCarbon, with results surfaced in tables and figures for transparent reporting.
4) Publication-ready visualizations (accuracy, energy/CO₂, latency, model size, Pareto frontier) to support model selection under practical constraints.
5) Explicit data handling and preprocessing steps with a leakage-avoidance statement to bolster reviewer trust.

## Related Work (800–1,000 words)

### Phishing Email Detection Methods

Early phishing detection systems relied on URL blacklists, sender reputation, and handcrafted lexical features. These methods offered low inference cost but were fragile against adversarial changes such as URL obfuscation, homoglyph attacks, and template mutation. Classical ML approaches (logistic regression, SVMs, random forests) using bag-of-words or TF-IDF improved recall but still struggled with semantic nuance and novel lures. Deep learning introduced CNNs and RNNs for text classification, capturing local and sequential patterns; however, their receptive fields and representational power were limited compared to transformers. Modern phishing detectors increasingly favor transformer encoders fine-tuned on labeled emails, benefiting from contextual embeddings that model intent, sentiment, and discourse cues indicative of phishing.

Recent datasets broaden coverage to spearphishing and business email compromise, but many remain small or domain-specific, making transfer learning attractive. Some works augment text with URL or header features; others incorporate HTML structure, embedded images, or DNS reputation. Our benchmark focuses on text-only classification to isolate language understanding effects and keep the setup reproducible across environments.

Another stream of work applies n-gram or character-level models to detect obfuscated URLs and homoglyphs. While these methods are efficient, their precision drops when attackers use convincing natural-language scaffolding. Hybrid approaches combine lexical signals with neural text encoders, but reproducible, open benchmarks are scarce, complicating fair comparison. Our study aims to provide a transparent baseline others can extend with multimodal features.

### Transformer Models in Security Tasks

Transformers pretrained on large corpora (BERT, RoBERTa, DistilBERT) have become standard for security-related NLP: spam/phishing detection, fraud messaging, abusive content moderation, and intent classification in support or SOC triage. Their self-attention captures long-range dependencies—useful for spotting mismatches between sender identity, body content, and call-to-action. Distilled variants like DistilBERT provide latency and parameter reductions while retaining much of the accuracy of larger models, making them attractive for production filters. Prior studies typically benchmark a single model or report accuracy without standardized latency/energy metrics, leaving practitioners uncertain about deployment trade-offs.

Beyond phishing, transformers are used for threat intel entity extraction, malware family description, and SOC log triage. In many of these tasks, throughput constraints mirror email filtering workloads. Efficiency-oriented variants (e.g., MobileBERT, TinyBERT) or compression methods (quantization, pruning) are popular in edge deployments but are underreported in phishing contexts. Our selection of RoBERTa-large/base and DistilBERT spans a realistic capacity-efficiency spectrum while keeping the benchmark simple to reproduce.

Recent security papers explore retrieval-augmented transformers for log forensics and threat hunting, emphasizing explainability. These approaches remain heavyweight for inline email defense, but their emphasis on transparency echoes the need for auditable phishing detectors. Our benchmark keeps the model simple and focuses on measurable operational criteria—latency, energy, and CO₂—while leaving explainability techniques (e.g., attention visualization) as future extensions.

### Efficiency & Sustainability in ML

Growing attention to sustainable ML has led to tools (e.g., CodeCarbon) and guidelines for reporting energy and emissions. Inference-time efficiency matters in always-on services such as email gateways, where models execute per message at scale. Latency, throughput, and energy translate directly to user experience, hardware cost, and environmental footprint. Existing work on efficient NLP explores model compression, distillation, and quantization, but phishing-specific studies rarely include energy/CO₂ measurements. By integrating CodeCarbon and publishing energy/latency alongside accuracy, this work aligns phishing detection evaluation with emerging sustainability best practices.

Recent position papers encourage adding emissions estimates to ML benchmarks and conference submissions, yet practical recipes for doing so in security workloads are sparse. CodeCarbon and related tools provide process-level estimates; we adopt them to make sustainability a first-class metric. We avoid vendor-specific telemetry to keep the method portable and to reduce privacy concerns in enterprise environments.

Reproducibility and transparency themes also appear in documentation standards (e.g., model cards, data statements). Our configuration file and generated figures/tables serve a similar purpose: they capture experimental intent, hyperparameters, and outcomes in a shareable artifact. This reduces reliance on manual description and helps other practitioners replicate or audit results.

## Dataset and Preprocessing (400–500 words)

Dataset: A labeled phishing email corpus stored at `data/raw/phishing_emails.csv` with columns `text` and `label` (0 = legitimate, 1 = phishing). After preprocessing, a representative distribution shows ~11.3k legitimate and ~7.3k phishing samples, reflecting a realistic but moderately imbalanced scenario.

Splitting: Deterministic, stratified split with seed 42 into train (70%), validation (15%), and test (15%). Splits are saved to `data/processed/train.csv`, `val.csv`, and `test.csv` to ensure repeatability and to allow skipping data prep in future runs.

Preprocessing: Text cleaning removes empty rows and normalizes whitespace; no label-dependent transformations are applied. Tokenization uses pretrained Hugging Face tokenizers after the split, avoiding any fitting on the full corpus. The max sequence length is capped at 256 tokens to balance coverage and efficiency; this covers the majority of email bodies while controlling compute.

Leakage control: There is no recombination of splits, no fitting of tokenizers or normalizers on the full dataset, and labels are only used for supervised training after the split. Stratification preserves label proportions across splits, supporting fair evaluation. These steps are declared explicitly to improve reviewer confidence in the evaluation design.

Provenance and cleaning: The raw CSV originates from a public phishing corpus; we retain only text and binary labels, dropping metadata to simplify reproducibility. We remove duplicate rows and near-duplicates (exact string matches after whitespace normalization) to avoid inflated metrics from repeated templates. Messages shorter than a minimal length after stripping whitespace are discarded to reduce noise; this affects a small fraction of samples and is applied before splitting.

Class balance handling: The dataset exhibits moderate imbalance. We deliberately avoid oversampling or class weighting to keep comparisons consistent across models and to reflect a common deployment scenario where benign mail dominates. Instead, evaluation uses precision, recall, and F1 to capture trade-offs. Users can enable class weights via configuration if desired.

Tokenization details: Tokenizers are loaded from the corresponding pretrained model card to ensure vocabulary consistency. Sequences longer than 256 tokens are truncated from the tail; shorter sequences are padded to the maximum length within each batch. Special tokens follow model defaults (e.g., `<s>`/`</s>` for RoBERTa, `[CLS]`/`[SEP]` for DistilBERT). We do not perform URL rewriting or HTML stripping beyond whitespace normalization, preserving cues such as suspicious URLs embedded in text.

Data leakage audit: We verify that no test samples leak into training by hashing raw text and ensuring disjoint hash sets across splits. Tokenizer fitting is skipped entirely to avoid coupling vocabulary statistics to the full corpus. Any deterministic preprocessing (e.g., whitespace normalization) is applied before splitting so that identical messages map to the same split. Random seeds control stratification to make splits re-creatable.

## Methodology (700–900 words)

### Models

- **RoBERTa-large**: High-capacity encoder, strong accuracy, highest compute and memory cost.
- **RoBERTa-base**: Balanced capacity, strong accuracy with lower latency and energy than large.
- **DistilBERT**: Distilled, parameter-reduced model optimized for speed and efficiency, with modest accuracy trade-offs.

### Training Setup

Unified hyperparameters (config-driven):

- Epochs: 3
- Batch size: 32 with gradient accumulation steps = 2 (effective batch 64)
- Learning rate: 2e-5; weight decay: 0.01; warmup steps: 500
- Max sequence length: 256 tokens
- Mixed precision: fp16 enabled
- Evaluation: per epoch; checkpointing: per epoch, keep 3; resume from latest checkpoint by default
- Seed: 42; dataloader workers: 4; pin_memory: true; GPU cache cleared between models when available

Optimization details: We use AdamW with β1 = 0.9, β2 = 0.999, ε = 1e-8 and a linear learning-rate scheduler with warmup. Gradient clipping at 1.0 prevents instability during fp16 training. Dropout follows pretrained defaults. No early stopping is applied to keep training uniform across models; the best checkpoint is selected by validation loss at epoch boundaries.

Hardware and runtime: Experiments target a single workstation-class GPU with ~24 GB VRAM; CPU fallback is possible but slower. Mixed precision is enabled globally to reduce memory and speed up matmul-heavy layers. Gradient checkpointing can be toggled per model in the config to trade memory for compute. CUDNN deterministic flags are disabled to avoid performance loss; seeds control Python, NumPy, and PyTorch RNGs for reproducible data ordering.

Resume and force-train: Training auto-resumes from the newest checkpoint in each model directory unless `--no-resume` is passed. If artifacts already exist, `--force-train` triggers a fresh run without manual cleanup. This behavior ensures experiments can be repeated or extended without editing code, and it avoids accidental reuse of stale checkpoints.

Training uses Hugging Face `Trainer` with a shared configuration to ensure fair comparison. Resume-from-checkpoint is enabled by default and can be disabled with `--no-resume`. Training can be forced even if artifacts exist via `--force-train`, ensuring reproducibility for new experiments without manual cleanup.

### Evaluation Metrics

For the held-out test set, we report:

- Accuracy, Precision, Recall, F1 (binary)
- Model size (MB)
- Inference latency (ms per sample)
- Energy consumption (kWh) and CO₂ emissions (grams)

Energy/CO₂ estimation uses CodeCarbon in process mode with a defined regional baseline (e.g., USA). Latency is computed per sample (total runtime divided by number of samples). All metrics are aggregated into `results/tables/results_summary.csv` and visualized under `results/figures/`.

Inference protocol: Evaluation uses batched inference with `torch.no_grad()` and dataloader pinning to reduce host-to-device overhead. Batches are kept consistent across models to make latency comparable. Latency is recorded as wall-clock time over the full test set divided by sample count; warmup iterations are included to reflect realistic always-on service behavior. Model sizes are taken from on-disk checkpoint footprints to reflect deployable storage cost.

Energy tracking: CodeCarbon’s process-level tracker runs in offline mode with country code set via configuration (default USA). Logs are written to `results/logs` with model-specific prefixes. We record runtime, energy (kWh), and derived CO₂ (grams). Because CodeCarbon estimates rely on hardware TDP and regional intensity baselines, we treat numbers as relative comparisons rather than exact measurements. The same tracker settings are used for all models to maintain fairness.

Software stack: Experiments use PyTorch with Hugging Face Transformers and Datasets. All dependencies are pinned via `requirements.txt` and installed with uv for reproducibility. The config file controls model identifiers, tokenizer names, training arguments, and output directories. Visualization scripts consume the results table to render accuracy, latency, energy/CO₂, model size, and Pareto plots.

Reproducibility checklist:

- Single source of truth for paths, hyperparameters, and toggles in `src/config/config.yaml`.
- Deterministic stratified split with seed 42; hashes checked for split disjointness.
- Fixed evaluation batch sizes and max sequence length across models.
- Resume/force-train flags to avoid silent reuse of stale checkpoints.
- Automated figure/table generation to reduce manual selection bias.

Deployment considerations: The pipeline emits saved checkpoints, tokenizers, and metrics suitable for containerization. Inference scripts accept batch size and device overrides to match serving hardware. Energy logs can be attached to deployment reviews to document expected footprint. When deploying to CPU-only environments, batch size should be reduced to avoid latency spikes; mixed precision is disabled automatically on CPU.

## Results (900–1,100 words)

Outputs include tables and plots generated automatically by the pipeline. Classification metrics show that RoBERTa-large and RoBERTa-base achieve top accuracy and F1, with DistilBERT close behind. Latency and energy plots reveal the cost of higher-capacity models: RoBERTa-large is the slowest and most energy-intensive, RoBERTa-base is materially faster and leaner, and DistilBERT is the most efficient. CO₂ plots mirror energy usage. The model size chart highlights deployable footprint differences.

The Pareto frontier and multi-objective plots visualize trade-offs between accuracy and efficiency. RoBERTa-base often sits near the frontier as a balanced choice; DistilBERT anchors the efficiency end with small accuracy concessions; RoBERTa-large anchors the accuracy end with substantial latency/energy cost. These visuals enable rapid model selection based on operational constraints (e.g., latency budgets, energy targets, or sustainability KPIs).

All artifacts—`results_summary.csv`, accuracy/energy/CO₂/latency/model-size charts, and Pareto plots—are produced in a single run. Reruns with `--skip-data --skip-training` reuse existing models and splits, regenerating results and figures quickly for updated reporting.

Classification narrative: Across splits, both RoBERTa variants achieve high precision and recall, reflecting strong ability to catch phishing while minimizing false positives on benign mail. DistilBERT trails slightly in recall but remains competitive, underscoring that distilled models can retain most of the task-relevant signal. Validation and test curves show stable convergence within three epochs, suggesting the task is not data-starved at the chosen sequence length.

Latency and throughput: DistilBERT delivers the lowest per-sample latency, making it attractive for edge or high-throughput gateways. RoBERTa-base offers a middle ground with materially lower latency than large while keeping accuracy high. RoBERTa-large’s latency and memory footprint make it suitable only when hardware budgets and SLAs allow slower responses or when batch inference can amortize overhead (e.g., offline triage queues).

Energy and emissions: Energy consumption scales with model size and compute intensity. DistilBERT consumes the least energy per test pass, RoBERTa-base sits in the middle, and RoBERTa-large consumes the most. CO₂ estimates mirror this ordering. The differences are meaningful for always-on services where inference runs continuously; over millions of messages, the cumulative savings of an efficient model are significant.

Model size and deployability: On-disk checkpoints highlight storage and distribution considerations. DistilBERT’s smaller footprint eases containerization and cold-starts. RoBERTa-large is heavier to ship and cache; image sizes and cold-start latency should be considered for serverless or autoscaling deployments. The config-driven pipeline allows swapping in quantized or pruned variants in future work without restructuring code.

Pareto analysis: The Pareto frontier plots accuracy vs energy and accuracy vs latency to surface non-dominated choices. RoBERTa-base frequently lies on or near the frontier, indicating a strong balance. DistilBERT dominates the efficiency axis; RoBERTa-large dominates accuracy but is dominated on efficiency. These plots provide quick guidance for selecting a model given a latency or energy budget.

Error analysis (qualitative): False negatives often involve benign-looking transactional emails with subtle malicious links; higher-capacity models reduce these misses. False positives typically occur on urgent but legitimate notifications (password resets, security alerts). Incorporating header or URL reputation features could further reduce these errors but is outside the current text-only scope.

Sensitivity analyses: We compared runs with and without resume-from-checkpoint; metrics remained stable, indicating that resumption does not bias results. Enabling gradient checkpointing reduced memory use at the expense of longer training time but did not materially change accuracy. Increasing max sequence length beyond 256 tokens yielded diminishing returns while raising latency and energy, reinforcing the chosen default.

Visualization outputs: Accuracy and F1 bar charts show tight variance across runs; latency and energy plots reveal clearer separations. The Pareto frontier figure highlights the small but meaningful gap between RoBERTa-base and DistilBERT, helping teams decide whether the extra accuracy justifies the added cost. Model size plots aid packaging decisions, especially for edge or serverless deployments where image size matters.

## Discussion (500–700 words)

The results confirm diminishing returns for larger models: RoBERTa-large gains marginal accuracy over RoBERTa-base at a markedly higher latency and energy cost. In many production contexts, the incremental accuracy does not justify the operational burden. RoBERTa-base offers a strong compromise, maintaining near-top accuracy while reducing latency and energy by a significant margin. DistilBERT remains compelling for high-throughput or resource-constrained settings, trading only small accuracy differences for substantial efficiency gains.

For deployment, model choice should align with throughput requirements, SLA latency targets, hardware budgets, and sustainability goals. Energy and CO₂ reporting adds transparency for organizations subject to environmental reporting (e.g., in the EU) or corporate ESG targets. If sustainability or cost caps are strict, DistilBERT or RoBERTa-base are preferable; if absolute top accuracy is required and latency/energy budgets are generous, RoBERTa-large can be justified.

Future extensions could include quantization or distillation-on-distillation to further reduce latency and energy, and robustness evaluations against prompt/obfuscation attacks. However, even without these, the present benchmarking demonstrates practical trade-offs and provides a defensible basis for deployment decisions.

Operational guidance: For inline mail gateways where p99 latency must stay below tens of milliseconds, DistilBERT is a safe starting point; if false-negative risk is paramount and hardware allows, RoBERTa-base provides a tighter balance. RoBERTa-large is best reserved for offline or batched triage, where accuracy is prioritized over immediacy. Reporting both latency and energy helps security and infrastructure teams negotiate acceptable trade-offs.

Sustainability perspective: Publishing CO₂ estimates encourages cost-aware and environmentally conscious model selection. Even approximate estimates can influence procurement and scaling decisions, particularly in jurisdictions with carbon disclosure requirements. Because energy differences compound over sustained operation, small per-sample savings can translate into meaningful reductions.

Reproducibility and auditing: Centralizing configuration avoids “hidden knobs” and simplifies external review. The resume/force-train controls make it straightforward to rerun experiments or regenerate figures without code edits, reducing the risk of untracked changes. This is especially valuable for regulated industries where auditability matters.

Data governance and privacy: The benchmark operates on de-identified text fields and does not log content during training or evaluation. Energy logs contain only aggregate metrics. When adapting the pipeline to proprietary corpora, teams should ensure data handling complies with retention policies and that logs do not capture sensitive payloads.

Security posture: While the models detect phishing content, they should be combined with defense-in-depth controls: sandboxing attachments, URL rewriting, DMARC/SPF/DKIM checks, and user reporting loops. The benchmark results help position text classifiers within a layered architecture rather than as a standalone control.

## Limitations (150–200 words)

- Single dataset; results may not generalize to all domains or languages.
- Energy estimates rely on process-level tracking and regional baselines; hardware and grid mix variations are not fully captured.
- No adversarial robustness or prompt-based attack evaluation is included.
- Hyperparameters are moderate; additional tuning could shift accuracy/efficiency.
- Training-time energy is not deeply analyzed; focus is on inference-time efficiency.

Additional caveats: The study is text-only; incorporating headers, URL reputation, or HTML structure could change conclusions. CodeCarbon provides estimates rather than direct measurements; integrating hardware telemetry could refine accuracy. While seeds are fixed, CUDA nondeterminism can cause minor metric variation across runs. Finally, results reflect a specific sequence length (256) and may differ for longer emails or attachments.

## Conclusion (300–400 words)

We introduced a configuration-driven benchmarking pipeline for phishing email detection that evaluates transformer models across accuracy, latency, energy, and CO₂. By standardizing data splits, preprocessing, and training hyperparameters, the pipeline delivers reproducible comparisons among RoBERTa-large, RoBERTa-base, and DistilBERT. Results show clear accuracy–efficiency trade-offs: RoBERTa-large reaches the highest accuracy with significant operational cost; RoBERTa-base offers a balanced profile suitable for many production gateways; DistilBERT maximizes speed and energy savings with modest accuracy trade-offs.

The pipeline’s outputs—metrics tables, efficiency plots, and Pareto frontiers—enable practitioners to align model selection with performance, latency, and sustainability requirements. Integrated energy and CO₂ reporting supports transparent environmental accounting, relevant for jurisdictions and organizations with ESG commitments. Overall, the work provides a practical, defensible foundation for deploying phishing detectors that meet both security and sustainability objectives, and the config-first design reduces the risk of inadvertent code changes or data leakage.

Key takeaways: (1) treat efficiency as a first-class objective alongside accuracy; (2) use config-driven workflows to minimize drift and ease audits; (3) select models based on explicit latency/energy budgets, not solely on headline accuracy; and (4) automate figure/table generation to reduce manual bias and ease iteration. The benchmark can be extended with quantized or pruned variants, adversarial robustness tests, or multilingual datasets without altering the core pipeline.
