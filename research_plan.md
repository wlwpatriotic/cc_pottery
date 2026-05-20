# Research Plan: Fine-Grained Painted Pottery Recognition via Generative Reasoning

## Target Venue: CVPR / ICCV / NeurIPS

## 1. PROBLEM FORMULATION

### Core Problem
Fine-grained recognition of ancient Chinese painted pottery across 79 archaeological cultures and 50 artifact types, with only 2,466 labeled samples. Standard deep learning approaches fail due to:
1. Long-tail class distribution (5 cultures >100 samples, 36 cultures <10 samples)
2. Fine inter-class visual differences (e.g., Majiayao vs Banshan pottery style)
3. Domain-specific visual patterns (decorative motifs, clay texture, firing marks)

### Key Insight
Archaeologists identify pottery through **compositional reasoning**: they observe decorative patterns, vessel shape, clay material, and then **reason step-by-step** about which culture and era produced it. This is fundamentally a **multimodal chain-of-thought** process that existing vision models cannot replicate.

## 2. PROPOSED METHOD: ArchaeoGPT

### Architecture Overview
1. **Hierarchical Vision Encoder**: Multi-scale feature extraction capturing motif-level, shape-level, and texture-level features
2. **Archaeological Knowledge-Augmented VLM**: Fine-tuned vision-language model with domain-specific archaeological terminology
3. **Generative Reasoning Head**: Produces structured CoT outputs: "This vessel has [motif pattern] typical of [culture], with [clay type] suggesting [era], therefore it is [culture] [type] from [era]"
4. **Open-Vocabulary Cultural Recognition**: CLIP-style contrastive learning with archaeological text descriptions

### Key Contributions
1. **First fine-grained pottery recognition dataset** with hierarchical labels (culture/type/era)
2. **Generative reasoning for FGVC**: Models that explain *why* they classify, not just predict
3. **Archaeological VLM**: Domain-adapted vision-language model for cultural heritage
4. **Cross-culture generalization**: Zero-shot recognition of rare cultures via compositional reasoning

## 3. EXPERIMENTAL DESIGN

### Dataset Splits
- Train: 70% (stratified by culture + type)
- Val: 15% (stratified)
- Test: 15% (stratified)
- Special split: "Open-culture" split where tail cultures (≤5 samples) held out for zero-shot evaluation

### Baselines
1. ResNet-50 / ViT-B (standard FGVC)
2. CLIP zero-shot (ViT-L/14)
3. DINOv2 fine-tuned
4. BLIP-2 / LLaVA-1.5 (general VLM)
5. PMA (Part-based FGVC method)
6. CAL (Counterfactual Attention Learning)

### Evaluation Metrics
- Top-1 / Top-5 Accuracy
- Per-class F1 (for long-tail analysis)
- Hierarchical Accuracy (culture→type→era)
- Open-vocabulary mAP
- Reasoning quality (GPT-4 judge for CoT correctness)

### Ablation Studies
1. Vision encoder (ResNet vs ViT vs DINOv2)
2. Text encoder (archaeological vs general-domain)
3. CoT reasoning depth (0/1/3/5 step)
4. Hierarchical vs flat classification
5. Data augmentation for long-tail
6. Cross-attention vs self-attention for vision-text fusion

## 4. IMPLEMENTATION ROADMAP

Phase 1: Data Pipeline
- Build PyTorch Dataset with hierarchical labels
- Image preprocessing and augmentation
- Train/val/test splits with stratification
- Text preprocessing for archaeological terminology

Phase 2: Baselines
- Implement and train standard baselines
- Zero-shot CLIP evaluation
- Report baseline results

Phase 3: ArchaeoGPT Core
- Hierarchical vision encoder
- Archaeological knowledge injection
- Generative reasoning head
- Training pipeline

Phase 4: Evaluation & Analysis
- Full benchmark evaluation
- Error analysis by culture/type
- Ablation studies
- Visualization dashboard

Phase 5: Paper Writing
- Figures and tables
- Method description
- Results analysis
- Reviewer rebuttal preparation
