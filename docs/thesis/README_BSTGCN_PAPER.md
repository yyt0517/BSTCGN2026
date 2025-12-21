# BSTGCN Paper - IEEE TIFS Submission

## 📄 Paper Information

**Title**: Bayesian Spatiotemporal Graph Neural Networks for Trustworthy Intrusion Detection in 5G/6G Networks

**Target Journal**: IEEE Transactions on Information Forensics and Security (TIFS)

**Status**: Initial Draft Completed

---

## 📁 Files

- `BSTGCN.tex` - Main LaTeX source file (论文主文件)
- `refs.bib` - Bibliography file (参考文献)
- `IEEEtran.cls` - IEEE journal class file (需要从IEEE官网下载)

---

## 🔧 Compilation Instructions

### Prerequisites

1. Install a LaTeX distribution:
   - **Windows**: MiKTeX or TeX Live
   - **macOS**: MacTeX
   - **Linux**: TeX Live

2. Ensure you have the following packages installed:
   - `IEEEtran` (IEEE journal class)
   - `amsmath`, `amsfonts` (mathematical symbols)
   - `algorithm`, `algorithmic` (algorithm formatting)
   - `graphicx` (figures)
   - `cite` (citations)

### Compilation Steps

#### Method 1: Using pdflatex (Recommended)

```bash
cd docs/thesis
pdflatex BSTGCN.tex
bibtex BSTGCN
pdflatex BSTGCN.tex
pdflatex BSTGCN.tex
```

#### Method 2: Using latexmk (Automated)

```bash
cd docs/thesis
latexmk -pdf BSTGCN.tex
```

#### Method 3: Using Overleaf

1. Upload `BSTGCN.tex` and `refs.bib` to Overleaf
2. Set compiler to `pdfLaTeX`
3. Click "Recompile"

---

## 📊 Paper Structure

### Current Sections (Complete)

1. **Abstract** (200 words)
   - Problem statement
   - Proposed method (BSTGCN)
   - Key results
   - Significance

2. **Introduction** (1.5 pages)
   - Motivation and challenges
   - Technical gaps
   - Our contributions (4 main points)

3. **Related Work** (1.5 pages)
   - Spatiotemporal GNNs for intrusion detection
   - Bayesian deep learning in security
   - Research gaps

4. **Methodology** (2.5 pages)
   - Problem formulation
   - BSTGCN architecture
   - Variational inference and training
   - Uncertainty quantification

5. **Experimental Setup** (1 page)
   - Dataset (5G-NIDD)
   - Baseline models (10+)
   - Implementation details
   - Evaluation metrics

6. **Results and Analysis** (1.5 pages)
   - Classification performance (Table 1)
   - Uncertainty quality (Table 2)
   - Per-class performance (Table 3)
   - OOD detection
   - Computational efficiency (Table 4)
   - Ablation study (Table 5)

7. **Discussion** (0.5 pages)
   - Why BSTGCN works
   - Implications for 6G security
   - Limitations and future work

8. **Conclusion** (0.5 pages)
   - Summary of contributions
   - Key results
   - Future directions

9. **Acknowledgments**

---

## ✅ TODO Before Submission

### Content

- [ ] Replace placeholder author names and affiliations (lines 20-25)
- [ ] Add funding information (line 23)
- [ ] Update manuscript dates (line 25)
- [ ] Add actual figures (currently placeholders):
  - [ ] Figure 1: BSTGCN architecture diagram
  - [ ] Figure 2: ST Block detailed structure
  - [ ] Figure 3: Uncertainty-error scatter plot
  - [ ] Figure 4: OOD detection AUROC curves
  - [ ] Figure 5: Per-class F1 comparison bar chart

### References

- [ ] Complete all reference entries in `refs.bib`
- [ ] Add missing citations:
  - [ ] E-GraphSAGE paper
  - [ ] BS-GAT paper
  - [ ] TGN paper
  - [ ] Additional baseline papers
- [ ] Verify all DOIs and page numbers

### Formatting

- [ ] Check all equations are numbered correctly
- [ ] Verify all tables are formatted per IEEE style
- [ ] Ensure all figures have captions
- [ ] Check citation format (use `\cite{}`)
- [ ] Proofread for typos and grammar

### Experimental Results

- [ ] Verify all numbers match your actual experimental results
- [ ] Update Table 1 with final classification results
- [ ] Update Table 2 with uncertainty metrics
- [ ] Update Table 3 with per-class F1 scores
- [ ] Update Table 4 with efficiency metrics
- [ ] Update Table 5 with ablation results

---

## 📈 Key Results to Highlight

### Main Results (from your experiments)

- **Accuracy**: 96.67% (vs. 95.97% for Hybrid baseline)
- **Macro F1**: 91.90% (vs. 89.73% for Hybrid baseline)
- **Uncertainty-Error Correlation**: 0.1777 (vs. 0.1156 for deterministic)
- **Validation Time Reduction**: 60% (30s → 12s per epoch)
- **OOD Detection AUROC**: 0.823 (temporal split)
- **Parameters**: 755K (52.1% in Bayesian ST blocks)

### Comparison with Baselines

| Model | Accuracy | Macro F1 | Unc-Err Corr |
|-------|----------|----------|--------------|
| CNN1D | 0.8881 | 0.8977 | N/A |
| GRU | 0.8678 | 0.8792 | N/A |
| TCN | 0.8222 | 0.8386 | N/A |
| LightGBM | 0.8865 | 0.8965 | N/A |
| GCN | 0.8706 | 0.8810 | N/A |
| GAT | 0.8747 | 0.8840 | N/A |
| LSTM+GNN | 0.8849 | 0.8941 | N/A |
| STGCN | 0.9349 | 0.9399 | N/A |
| Hybrid | 0.9597 | 0.8973 | 0.1156 |
| BCNN-IDS | 0.9120 | 0.8690 | 0.1534 |
| **BSTGCN** | **0.9667** | **0.9190** | **0.1777** |

---

## 📝 Writing Tips

1. **Be concise**: IEEE TIFS typically allows 8-10 pages for regular papers
2. **Use active voice**: "We propose BSTGCN" instead of "BSTGCN is proposed"
3. **Quantify claims**: Always provide numbers to support statements
4. **Compare fairly**: Ensure all baselines use the same data split and hyperparameters
5. **Highlight novelty**: Emphasize what makes BSTGCN different from existing work

---

## 🔗 Useful Links

- [IEEE TIFS Author Guidelines](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=10206)
- [IEEE LaTeX Template](https://www.ieee.org/conferences/publishing/templates.html)
- [Overleaf IEEE Template](https://www.overleaf.com/latex/templates/ieee-journal-paper-template/ypypvwjmvtdf)

---

## 📧 Contact

For questions about this paper, please contact:
- [Your Email]
- [Supervisor Email]

---

**Last Updated**: 2025-01-26

