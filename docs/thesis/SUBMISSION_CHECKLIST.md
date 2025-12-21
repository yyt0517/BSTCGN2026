# IEEE TIFS Submission Checklist

## 📋 Pre-Submission Checklist

### Content Completeness

#### Abstract & Keywords
- [ ] Abstract is 150-250 words
- [ ] Abstract includes: problem, method, results, significance
- [ ] Keywords are relevant and specific (5-8 keywords)
- [ ] Keywords match IEEE Xplore taxonomy

#### Introduction
- [ ] Clearly states the problem and motivation
- [ ] Explains why existing solutions are insufficient
- [ ] Lists specific contributions (numbered list)
- [ ] Provides paper organization at the end

#### Related Work
- [ ] Covers all relevant prior work
- [ ] Clearly identifies research gaps
- [ ] Explains how your work differs from existing approaches
- [ ] Cites recent papers (2020-2025)

#### Methodology
- [ ] Problem formulation is mathematically rigorous
- [ ] All notation is defined before use
- [ ] Architecture is clearly described
- [ ] All equations are numbered and referenced
- [ ] Algorithm pseudocode is provided where appropriate

#### Experiments
- [ ] Dataset is described in detail (size, classes, splits)
- [ ] All baseline models are clearly described
- [ ] Hyperparameters are specified
- [ ] Evaluation metrics are defined
- [ ] Implementation details are provided (hardware, software, training time)

#### Results
- [ ] All tables have captions above the table
- [ ] All figures have captions below the figure
- [ ] All tables and figures are referenced in text
- [ ] Results are compared with all baselines
- [ ] Statistical significance is discussed where appropriate
- [ ] Ablation study validates each component

#### Discussion
- [ ] Explains why the method works
- [ ] Discusses limitations honestly
- [ ] Suggests future research directions
- [ ] Relates findings to broader implications

#### Conclusion
- [ ] Summarizes key contributions
- [ ] Highlights main results
- [ ] Avoids introducing new information

---

### Figures and Tables

#### Figures (5 required)
- [ ] Figure 1: BSTGCN architecture diagram
  - Shows: Input → Transformer → KNN Graph → ST Blocks → Classifier
  - High resolution (300+ DPI)
  - Clear labels and legends
  
- [ ] Figure 2: ST Block detailed structure
  - Shows: Bayesian TCN + Bayesian GraphSAGE + Gated Fusion
  - Annotates key parameters
  
- [ ] Figure 3: Uncertainty-error scatter plot
  - X-axis: Epistemic Uncertainty
  - Y-axis: Prediction Error
  - Shows correlation
  
- [ ] Figure 4: OOD detection ROC curves
  - Compares BSTGCN vs. baselines
  - Labels AUROC values
  
- [ ] Figure 5: Per-class F1 comparison
  - Bar chart for 9 classes
  - Compares 3 models (BSTGCN, Hybrid, BCNN-IDS)

#### Tables (5 included)
- [ ] Table 1: Classification performance (all models)
- [ ] Table 2: Uncertainty quantification quality
- [ ] Table 3: Per-class F1 scores (minority classes)
- [ ] Table 4: Computational efficiency
- [ ] Table 5: Ablation study results

#### Figure/Table Quality
- [ ] All figures are vector graphics (PDF/EPS) or high-res (300+ DPI)
- [ ] All text in figures is readable (font size ≥ 8pt)
- [ ] All tables use IEEE three-line format (top, middle, bottom)
- [ ] All captions are descriptive and self-contained
- [ ] All figures/tables fit within column width

---

### References

#### Citation Completeness
- [ ] All claims are supported by citations
- [ ] All baseline models are cited
- [ ] Dataset paper is cited
- [ ] All methods used (Transformer, GraphSAGE, TCN) are cited
- [ ] Recent related work (2023-2025) is cited

#### Citation Format
- [ ] All references use IEEE format
- [ ] All journal papers include: authors, title, journal, volume, pages, year
- [ ] All conference papers include: authors, title, conference, pages, year
- [ ] All DOIs are included where available
- [ ] All URLs are valid and accessible

#### Reference Quality
- [ ] At least 30 references (typical for IEEE TIFS)
- [ ] Mix of journals (60%) and conferences (40%)
- [ ] Includes recent papers (50% from last 5 years)
- [ ] Cites IEEE TIFS papers (shows familiarity with venue)

---

### Writing Quality

#### Language
- [ ] No grammatical errors (use Grammarly or similar)
- [ ] No spelling mistakes
- [ ] Consistent terminology throughout
- [ ] Active voice used where appropriate
- [ ] Technical terms are defined on first use

#### Style
- [ ] Follows IEEE style guide
- [ ] Uses past tense for experiments ("We trained...")
- [ ] Uses present tense for general facts ("GNNs are...")
- [ ] Avoids colloquialisms and informal language
- [ ] Avoids prohibited words (per your style guide)

#### Clarity
- [ ] Each paragraph has a clear topic sentence
- [ ] Transitions between sections are smooth
- [ ] Complex concepts are explained clearly
- [ ] Acronyms are defined on first use
- [ ] Notation is consistent throughout

---

### Formatting

#### IEEE Template
- [ ] Uses official IEEE LaTeX template
- [ ] Document class: `\documentclass[lettersize,journal]{IEEEtran}`
- [ ] Correct page size (US Letter)
- [ ] Two-column format
- [ ] Correct margins and spacing

#### Equations
- [ ] All equations are numbered (except very simple ones)
- [ ] Equation numbers are on the right
- [ ] Equations are referenced as "Eq. (1)" or "(1)"
- [ ] Multi-line equations use `align` or `multline`
- [ ] Variables are italicized, functions are upright

#### Sections
- [ ] Section titles use title case
- [ ] Sections are numbered (I, II, III, ...)
- [ ] Subsections are numbered (A, B, C, ...)
- [ ] No orphan headings (heading at bottom of column)

---

### Author Information

#### Author Details
- [ ] All author names are correct and complete
- [ ] Author affiliations are accurate
- [ ] Corresponding author is indicated
- [ ] Email addresses are provided
- [ ] ORCID IDs are included (if available)

#### Funding
- [ ] All funding sources are acknowledged
- [ ] Grant numbers are provided
- [ ] Funding statement is in acknowledgments

#### Conflicts of Interest
- [ ] Any conflicts of interest are disclosed
- [ ] If no conflicts, state "The authors declare no conflicts of interest"

---

### Supplementary Materials

#### Code and Data
- [ ] Code repository URL is provided (if applicable)
- [ ] Dataset availability is stated
- [ ] Reproducibility information is included

#### Supplementary Files
- [ ] Supplementary material is prepared (if needed)
- [ ] Supplementary material is referenced in main text

---

### Final Checks

#### Compilation
- [ ] LaTeX compiles without errors
- [ ] All references are resolved (no "??" in PDF)
- [ ] All citations are resolved
- [ ] PDF is generated successfully

#### Page Limit
- [ ] Paper is within page limit (typically 8-10 pages for regular papers)
- [ ] If over limit, consider moving content to supplementary material

#### Plagiarism
- [ ] Run through plagiarism checker (Turnitin, iThenticate)
- [ ] Self-plagiarism is avoided (if reusing your own text)
- [ ] All borrowed text is properly quoted and cited

#### Proofreading
- [ ] Read entire paper aloud
- [ ] Have colleague/advisor review
- [ ] Check all numbers match experimental results
- [ ] Verify all cross-references are correct

---

## 📤 Submission Process

### IEEE TIFS Submission

1. **Create IEEE Account**
   - [ ] Register at https://mc.manuscriptcentral.com/t-ifs
   - [ ] Complete author profile

2. **Prepare Submission Files**
   - [ ] Main manuscript (PDF)
   - [ ] Source files (LaTeX + figures)
   - [ ] Cover letter
   - [ ] Supplementary materials (if any)

3. **Cover Letter Should Include**
   - [ ] Brief summary of paper (2-3 sentences)
   - [ ] Why it's suitable for IEEE TIFS
   - [ ] Confirmation of originality
   - [ ] Suggested reviewers (3-5)
   - [ ] Excluded reviewers (if any)

4. **Submission Metadata**
   - [ ] Title
   - [ ] Abstract
   - [ ] Keywords
   - [ ] Author information
   - [ ] Suggested associate editors

5. **Final Submission**
   - [ ] Upload all files
   - [ ] Review submission preview
   - [ ] Submit manuscript
   - [ ] Save submission confirmation

---

## ⏰ Timeline

- **Week 1**: Complete experiments and figures
- **Week 2**: Finalize references and formatting
- **Week 3**: Internal review and revisions
- **Week 4**: Final proofreading and submission

---

## 📧 Contact

If you have questions about the submission process:
- IEEE TIFS Editorial Office: t-ifs@ieee.org
- IEEE Author Support: https://ieeeauthorcenter.ieee.org/

---

**Good luck with your submission! 🎉**

