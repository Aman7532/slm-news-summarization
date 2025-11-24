# Evaluating Small Language Models for News Summarization

**Implications and Factors Influencing Performance**

[![Paper](https://img.shields.io/badge/Paper-LaTeX-blue)](research_paper.tex)
[![Presentation](https://img.shields.io/badge/Presentation-LaTeX-green)](research_presentation.tex)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

This repository contains the complete implementation and analysis of our research on **Small Language Models (SLMs)** for news summarization. We comprehensively evaluate three popular SLMs across four different learning approaches to understand which strategies work best for different model architectures.

### Research Questions

1. **RQ1**: How do different learning approaches affect performance across SLM architectures?
2. **RQ2**: Does fine-tuning improve or degrade instruction-tuned models?
3. **RQ3**: What are the minimum model size requirements for effective LoRA fine-tuning?
4. **RQ4**: How do performance-efficiency trade-offs vary across learning approaches?

### Models Evaluated

- **FLAN-T5-Small** (80M parameters) - Instruction-tuned
- **FLAN-T5-Base** (250M parameters) - Instruction-tuned
- **BART-Base** (140M parameters) - Generation-focused

### Learning Approaches

- **Zero-shot**: Direct inference without training
- **Few-shot**: In-context learning with 3 examples
- **Full Fine-tuning**: Complete model parameter updates
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning

### Dataset

- **CNN/DailyMail**: 1,000 training samples, 100 test samples
- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore

---

## 🏆 Key Findings

### 1. **Catastrophic Forgetting in Instruction-Tuned Models** ⚠️

Fine-tuning **degrades** instruction-tuned models:
- FLAN-T5-Base: 32.50% (zero-shot) → 28.39% (fine-tuned) **[-4.11 pp]**
- FLAN-T5-Small: 29.56% (zero-shot) → 28.39% (fine-tuned) **[-1.17 pp]**

In contrast, generation-focused models benefit from fine-tuning:
- BART-Base: 30.74% (zero-shot) → 33.06% (fine-tuned) **[+2.32 pp]**

### 2. **LoRA Size Thresholds** 📊

LoRA effectiveness depends on model size and architecture:

| Model | Size | Architecture | LoRA vs Full FT | Status |
|-------|------|--------------|-----------------|--------|
| FLAN-T5-Small | 80M | Instruction | -2.23 pp | ❌ Fails |
| BART-Base | 140M | Generation | -1.93 pp | ⚠️ Acceptable |
| FLAN-T5-Base | 250M | Instruction | +1.36 pp | ✅ Success |

**Threshold**: 250M+ for instruction-tuned, 140M+ for generation-focused models

### 3. **Zero-Shot Superiority** 🎯

Zero-shot achieves **highest average performance** across all models:

| Approach | Average ROUGE-1 | Rank |
|----------|-----------------|------|
| Zero-shot | **30.93%** | 🥇 1st |
| Full Fine-tuning | 29.95% | 2nd |
| LoRA Fine-tuning | 29.01% | 3rd |
| Few-shot | 23.97% | 4th |

**Implication**: Training doesn't always help!

### 4. **Few-Shot Unreliability** ⚠️

Few-shot learning shows extreme variability (20.3 pp range):
- FLAN-T5-Small: 32.29% ✅
- FLAN-T5-Base: 27.63% ⚠️
- BART-Base: 11.99% ❌ (catastrophic failure)

**Conclusion**: Too risky for production systems

### 5. **LoRA Efficiency Benefits** ⚡

For BART-Base, LoRA provides excellent trade-offs:
- **Performance**: 94% of full fine-tuning (31.13% vs 33.06%)
- **Training Time**: 50% faster (20-30 min vs 45-60 min)
- **Parameters Trained**: 99.6% fewer (0.5M vs 140M)
- **Memory Usage**: 40% less (6-8 GB vs 10-12 GB)

### 6. **Architecture-Approach Matching** 🏗️

**Instruction-Tuned Models (FLAN-T5)**:
- ✅ Recommended: Zero-shot, Few-shot
- ❌ Not Recommended: Full fine-tuning

**Generation-Focused Models (BART)**:
- ✅ Recommended: Full fine-tuning, LoRA fine-tuning
- ❌ Not Recommended: Few-shot

---

## 📁 Repository Structure

```
.
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── research_paper.tex                     # LaTeX research paper
├── research_paper_complete.zip            # Paper + figures (ready for Overleaf)
│
├── research_presentation.tex              # LaTeX presentation (27 slides)
├── research_presentation_complete.zip     # Presentation + figures (ready for Overleaf)
│
├── flan_t5_small_results.json            # All results for FLAN-T5-Small
├── flan_t5_base_results.json             # All results for FLAN-T5-Base
├── bart_base_results.json                # All results for BART-Base
│
├── flan_t5_small.ipynb                   # Zero-shot + Few-shot (local)
├── flan_t5_base.ipynb                    # Zero-shot + Few-shot (local)
├── bart_base.ipynb                       # Zero-shot + Few-shot (local)
│
├── flan_t5_small_finetune_kaggle.ipynb   # Full fine-tuning (Kaggle GPU)
├── flan_t5_base_finetune_kaggle.ipynb    # Full fine-tuning (Kaggle GPU)
├── bart_base_finetune_kaggle.ipynb       # Full fine-tuning (Kaggle GPU)
│
├── flan_t5_small_lora_finetune_kaggle.ipynb  # LoRA fine-tuning (Kaggle GPU)
├── flan_t5_base_lora_finetune_kaggle.ipynb   # LoRA fine-tuning (Kaggle GPU)
└── bart_base_lora_finetune_kaggle.ipynb      # LoRA fine-tuning (Kaggle GPU)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (for fine-tuning)
- 8GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/slm-news-summarization.git
cd slm-news-summarization
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
evaluate>=0.4.0
rouge-score>=0.1.2
bert-score>=0.3.13
peft>=0.4.0
accelerate>=0.20.0
```

---

## 💻 Usage

### Running Zero-Shot and Few-Shot Evaluations (Local)

```bash
# FLAN-T5-Small
jupyter notebook flan_t5_small.ipynb

# FLAN-T5-Base
jupyter notebook flan_t5_base.ipynb

# BART-Base
jupyter notebook bart_base.ipynb
```

**Requirements**: CPU/GPU with 8GB+ RAM, ~30 minutes per notebook

### Running Fine-Tuning (Kaggle GPU)

1. Upload the respective `*_finetune_kaggle.ipynb` to Kaggle
2. Enable GPU accelerator (Settings → Accelerator → GPU)
3. Run all cells
4. Download results

**Requirements**: Kaggle GPU, ~45-60 minutes per notebook

### Running LoRA Fine-Tuning (Kaggle GPU)

1. Upload the respective `*_lora_finetune_kaggle.ipynb` to Kaggle
2. Enable GPU accelerator
3. Run all cells
4. Download results

**Requirements**: Kaggle GPU, ~20-30 minutes per notebook

---

## 📊 Results

### Top 3 Performers

| Rank | Model + Approach | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|------|------------------|---------|---------|---------|-----------|
| 🥇 | **BART-Base Full FT** | **33.06%** | 12.39% | **23.06%** | **87.55%** |
| 🥈 | **FLAN-T5-Base Zero-shot** | **32.50%** | **12.93%** | **22.85%** | **87.19%** |
| 🥉 | **FLAN-T5-Small Few-shot** | **32.29%** | **12.90%** | **22.71%** | **87.68%** |

### Complete Results

All detailed results are available in JSON format:
- [`flan_t5_small_results.json`](flan_t5_small_results.json)
- [`flan_t5_base_results.json`](flan_t5_base_results.json)
- [`bart_base_results.json`](bart_base_results.json)

### Visualizations

Generate figures from results:
```python
# Figures are included in the ZIP files
# To regenerate: see research_paper.tex and research_presentation.tex
```

---

## 📄 Research Paper & Presentation

### Compile Paper (Overleaf)

1. Go to [Overleaf](https://www.overleaf.com)
2. Upload `research_paper_complete.zip`
3. Click "Recompile"
4. Download PDF

**Output**: 8-10 page two-column academic paper

### Compile Presentation (Overleaf)

1. Go to [Overleaf](https://www.overleaf.com)
2. Upload `research_presentation_complete.zip`
3. Click "Recompile"
4. Download PDF

**Output**: 27-slide professional presentation

---

## 🎯 Practical Recommendations

### For Practitioners

**Scenario-Based Recommendations**:

| Scenario | Best Choice | ROUGE-1 | Training |
|----------|-------------|---------|----------|
| No training data | FLAN-T5-Base Zero-shot | 32.50% | None |
| Max performance | BART-Base Full FT | 33.06% | 45-60 min |
| Efficiency focus | BART-Base LoRA FT | 31.13% | 20-30 min |
| Limited compute | FLAN-T5-Small Few-shot | 32.29% | None |
| Multiple tasks | FLAN-T5-Base Zero-shot | 32.50% | None |

### Decision Framework

```
Do you have training data?
├─ NO  → Use FLAN-T5-Base Zero-shot (32.50%)
│
└─ YES → Is model instruction-tuned?
         ├─ YES → Use Zero-shot (don't fine-tune!)
         └─ NO  → Need max performance?
                  ├─ YES → Full fine-tuning
                  └─ NO  → LoRA fine-tuning
```

---


## 👥 Authors

- **Aman Agarwal** - International Institute of Information Technology, Bangalore
- **Nakul Siwach** - International Institute of Information Technology, Bangalore
- **Himanshu Shivhare** - International Institute of Information Technology, Bangalore

---

## 🙏 Acknowledgments

- Reference paper: Xu et al., "Evaluating Small Language Models for News Summarization"
- Hugging Face Transformers library
- CNN/DailyMail dataset creators
- Kaggle for GPU resources

---


## 🐛 Issues & Contributions

Found a bug or want to contribute? Please open an issue or submit a pull request!

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📮 Contact

For questions or collaborations:
- Open an issue in this repository
- Email: aman.agarwal@iiitb.ac.in

---

## 🔗 Links

- [Presentation (LaTeX)](research_presentation.tex)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

---

## 📊 Project Statistics

- **Models Evaluated**: 3
- **Learning Approaches**: 4
- **Total Experiments**: 12
- **Training Samples**: 1,000
- **Test Samples**: 100
- **Novel Findings**: 6 major contributions
- **Paper Pages**: 8-10 (two-column format)
- **Presentation Slides**: 27

---

**⭐ Star this repository if you find it helpful!**

**🔄 Fork it to use it for your own research!**

---

Last Updated: November 2024

