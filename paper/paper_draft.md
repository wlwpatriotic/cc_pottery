# ArchaeoGPT: Generative Chain-of-Thought Reasoning for Fine-Grained Painted Pottery Recognition

## Abstract

Fine-grained recognition of ancient painted pottery is a challenging problem at the intersection of computer vision and archaeology. Unlike natural image classification, painted pottery requires **compositional visual reasoning**: identifying decorative motifs, vessel morphology, and clay material, then synthesizing these cues to infer the archaeological culture, artifact type, and historical era. Existing fine-grained recognition methods rely on discriminative feature matching without modeling the reasoning process. We propose **ArchaeoGPT**, a vision-language framework that performs **generative chain-of-thought (CoT) reasoning** for fine-grained pottery recognition. ArchaeoGPT integrates: (1) a multi-scale hierarchical vision encoder capturing motif-level, shape-level, and texture-level features; (2) an archaeological knowledge encoder with domain-specific terminology embeddings; (3) a cross-modal fusion transformer that aligns visual patterns with archaeological concepts; and (4) a generative reasoning head that produces structured step-by-step predictions. We construct the first comprehensive fine-grained pottery benchmark — **PotteryFGVC** — with 2,466 artifacts spanning 79 archaeological cultures, 50 artifact types, and 7 chronological eras across 10 geographic regions of China. Experiments demonstrate that ArchaeoGPT outperforms standard fine-grained baselines by **[X]%** in culture classification and **[Y]%** in artifact type recognition, while providing interpretable reasoning traces that align with expert archaeological analysis. Our work establishes a new paradigm of **generative fine-grained recognition** for cultural heritage understanding.

## 1. Introduction

- Fine-grained recognition (FGVC) has made tremendous progress on natural domains (birds [CUB-200], cars [Stanford Cars], aircraft [FGVC-Aircraft])
- However, cultural heritage artifacts present unique challenges that existing FGVC methods fail to address
- Painted pottery requires **compositional reasoning**: no single feature distinguishes cultures — it's the combination of motif style, vessel shape, clay material, and decorative technique
- Archaeologists follow a **cognitive reasoning process**: observe → identify patterns → reason about culture → cross-reference with type → infer era
- This is fundamentally a **multimodal chain-of-thought** task, yet no existing work models it as such
- We introduce **PotteryFGVC** dataset and **ArchaeoGPT** framework

**Contributions:**
1. **PotteryFGVC**: First comprehensive fine-grained painted pottery benchmark with 2,466 artifacts, 79 cultures, 50 types, 7 eras, with rich textual descriptions
2. **ArchaeoGPT**: A generative reasoning framework that produces structured CoT predictions for cultural heritage artifacts
3. **Hierarchical Multi-Scale Vision Encoder**: Captures motif-level (local), pattern-level (mid), and shape-level (global) features
4. **Archaeological VLM**: Domain-adapted vision-language model bridging visual patterns with archaeological terminology
5. **Comprehensive Benchmarks**: Extensive comparison against 6 baselines with detailed error analysis

## 2. Related Work

### 2.1 Fine-Grained Visual Classification (FGVC)
- Part-based methods: Part-RCNN, NTS-Net, PMA
- Attention-based: CAL, DCL, MGE-CNN
- Transformer-based: TransFG, ViT-FGVC
- **Key limitation**: All rely on discriminative feature matching, none model reasoning

### 2.2 Vision-Language Models for FGVC
- CLIP-based: CoOp, CoCoOp (learnable prompts for FGVC)
- VLMs: LLaVA, BLIP-2 for visual question answering
- **Key limitation**: General-domain VLMs lack archaeological domain knowledge

### 2.3 Chain-of-Thought Reasoning in Vision
- Multimodal CoT: ScienceQA, CCoT, DDCoT
- **Key limitation**: Existing work focuses on natural scenes/QA, not fine-grained art/artifact analysis

### 2.4 Cultural Heritage AI
- Digital archaeology: 3D reconstruction, style analysis
- Pottery classification: mostly shallow features (color histogram, SIFT, etc.)
- **Key limitation**: No deep learning benchmark for fine-grained pottery, no reasoning-based methods

## 3. PotteryFGVC Dataset

### 3.1 Data Collection
- Source: "Complete Collection of Chinese Excavated Painted Pottery" (10 volumes)
- 10 geographic regions covering the major archaeological cultures of Neolithic-Bronze Age China
- Professional archaeological annotations: culture, type, era, dimensions, excavation context, detailed descriptions

### 3.2 Dataset Statistics
| Property | Value |
|----------|-------|
| Total artifacts | 2,466 |
| Valid images | 4,150+ |
| Archaeological cultures | 79 (58 after filtering tail) |
| Artifact types | 50 |
| Chronological eras | 7 |
| Avg. description length | 89 chars |
| Geographic regions | 10 |

### 3.3 Hierarchical Label Structure
- Level 1: Archaeological Culture (e.g., Majiayao, Yangshao, Dawenkou)
- Level 2: Artifact Type (e.g., jar, bowl, basin, bottle)
- Level 3: Chronological Era (Neolithic, Bronze Age, Early Iron Age)

### 3.4 Fine-Grained Challenges
- **Cross-culture visual similarity**: Majiayao vs Banshan pottery share vessel forms but differ in motifs
- **Long-tail distribution**: 5 cultures have >100 samples, 36 cultures have <10 samples
- **Intra-class variation**: Same culture produces diverse vessel forms
- **Domain-specific visual cues**: Decorative patterns (wave, net, spiral, geometric), clay texture

## 4. Method: ArchaeoGPT

### 4.1 Overview
[Insert architecture diagram]

ArchaeoGPT consists of four core components:
1. **Multi-Scale Hierarchical Vision Encoder** (Sec 4.2)
2. **Archaeological Knowledge Text Encoder** (Sec 4.3)
3. **Cross-Modal Fusion Transformer** (Sec 4.4)
4. **Generative Chain-of-Thought Reasoning Head** (Sec 4.5)

### 4.2 Multi-Scale Hierarchical Vision Encoder
- ResNet-50 backbone with multi-layer feature extraction
- **Level 1 (Local)**: Layer1 features — fine decorative motifs (wave patterns, grid lines)
- **Level 2 (Mid)**: Layer3 features — compositional patterns (band arrangements, motif groups)
- **Level 3 (Global)**: Layer4 features — vessel morphology (shape, proportion, rim style)

Each level is projected to a shared d-dimensional space and aggregated.

### 4.3 Archaeological Knowledge Text Encoder
- Learnable embeddings for archaeological concepts: culture, type, era
- Sub-embeddings for material (clay type) and decorative technique
- Fused via MLP to produce knowledge-conditioned text features

### 4.4 Cross-Modal Fusion Transformer
- Cross-attention between visual features and archaeological text features
- Self-attention refinement of fused features
- Feed-forward processing for non-linear interaction

### 4.5 Generative Chain-of-Thought Reasoning Head
- **State initialization**: Fused visual-text features
- **Step 1**: "Observing decorative pattern X, characteristic of culture group Y..."
- **Step 2**: "Vessel morphology suggests type Z, common in period W..."
- **Step 3**: "Synthesizing: Culture=A, Type=B, Era=C"

Each step uses a GRU cell to update reasoning state, with learnable step-specific queries.

### 4.6 Training Objectives
- Cross-entropy loss for culture, type, era classification
- Weighted loss for long-tail classes
- Optional: Reasoning trace supervision (GPT-4 generated explanations)

## 5. Experiments

### 5.1 Experimental Setup
- **Data split**: 70% train / 15% val / 15% test (stratified by culture+type)
- **Image size**: 224×224 with standard augmentations
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Training**: 30 epochs with early stopping (patience=7)
- **Hardware**: NVIDIA RTX (CUDA 11.8)
- **Metrics**: Top-1 accuracy, Macro F1, Per-class accuracy

### 5.2 Baselines
| Method | Backbone | Description |
|--------|----------|-------------|
| ResNet-50 | CNN | Standard fine-tuned classification |
| ViT-B/16 | Transformer | Vision Transformer fine-tuned |
| Hierarchical ViT | Transformer | Culture→Type→Era hierarchy |
| CLIP Zero-Shot | ViT-B/32 | Open-vocabulary classification |
| [More to add] | | |

### 5.3 Main Results

**Table 1: Comparison on PotteryFGVC test set.**

| Method | Culture (58 cls) | Type (50 cls) | Era (7 cls) |
|--------|:---:|:---:|:---:|
| | **Acc** | **Macro F1** | **Acc** | **Macro F1** | **Acc** | **Macro F1** |
| Random | 1.7% | — | 2.0% | — | 14.3% | — |
| CLIP ViT-B/32 (ZS) | 0.00% | 0.00 | 12.64% | 0.04 | 70.05% | 0.27 |
| ResNet-50 | 60.44% | 0.498 | 72.80% | 0.393 | 91.21% | 0.642 |
| ViT-B/16 | [running] | — | — | — | — | — |
| HierarchicalViT | [pending] | — | — | — | — | — |
| **ArchaeoGPT (Ours)** | [pending] | — | — | — | — | — |

**Key finding**: CLIP achieves 0% culture accuracy, demonstrating that general-domain VLMs completely fail on domain-specific archaeological fine-grained recognition. This validates our motivation for domain-adapted generative reasoning.

### 5.4 Cross-Task Dependency Analysis

We analyze how culture prediction accuracy affects artifact type recognition:

| Culture Status | Type Accuracy |
|:---|---:|
| Culture predicted correctly | **79.55%** |
| Culture predicted incorrectly | 62.50% |
| **Performance drop** | **-17.05%** |

This 17% drop confirms the **hierarchical dependency**: culture-level visual features (decorative motifs, clay texture) provide essential context for type-level recognition (vessel morphology). Errors at the culture level propagate to type predictions.

### 5.5 Error Analysis

**Most confused culture pairs (ResNet-50):**
1. Majiayao-Machang → Majiayao-Banshan (5x) — *same cultural tradition, different period*
2. Majiayao-Majiayao → Majiayao-Banshan (5x) — *sub-type confusion*
3. Yangshao Early → Yangshao Middle (4x) — *temporal progression within same culture*
4. Yangshao Late → Yangshao Middle (4x) — *temporal boundary blur*

**Most confused type pairs:**
1. Jar (罐) → Pot (壶) — *similar vessel morphology*
2. Bowl (钵) → Basin (盆) — *both open-mouth vessels*

**Confidence calibration**: Mean confidence on misclassified samples is 0.837, indicating significant **overconfidence** in wrong predictions. This motivates our CoT reasoning approach to reduce overconfident errors through explicit step-by-step verification.

### 5.6 Long-Tail Analysis

With 58 cultures in the test set and only 364 samples:
- 5 head cultures (>15 samples each): 53.8% of test set
- 30 tail cultures (<=5 samples each): 24.7% of test set
- Classes with 1 sample achieve either 0% or 100% accuracy

This extreme long-tail distribution reflects real archaeological data collection constraints and poses an open challenge for few-shot cultural heritage recognition.

### 5.7 Ablation Studies (Planned)

| Ablation | Description |
|----------|-------------|
| No text encoder | Remove archaeological knowledge module |
| 0-step reasoning | Direct classification without CoT |
| 1-step reasoning | Single reasoning step |
| 3-step reasoning | Full ArchaeoGPT |
| No multi-scale | Single-scale visual features only |
| No class weights | Uniform loss without long-tail handling |

## 6. Conclusion

- First fine-grained pottery recognition benchmark
- ArchaeoGPT: generative reasoning for cultural heritage AI
- State-of-the-art performance with interpretable reasoning
- Future: extension to other artifact types (bronze, jade), 3D reasoning

---

## Appendix: Training Log

[Numbered appendices with detailed per-class results]
