# Interpretable Deep-Fuzzy Skin Lesion Classification

Hybrid and interpretable framework for skin cancer classification that combines
**deep learning**, **Grad-CAM visual attention**, and **fuzzy logic** to unify
predictive performance with clinical transparency.

Official implementation of *"A Multi-Level Interpretable Framework for Skin
Cancer Classification Using Deep Learning and Fuzzy Systems"* (BRACIS 2025).

---

## Overview

Most deep learning models for skin lesion classification operate as "black
boxes": they return a prediction without explaining which visual features
supported the decision. This project addresses that gap with a three-stage
pipeline:

1. **EfficientNet-B0** (fine-tuned on HAM10000) estimates the probability of
   malignancy, `P(malignant)`.
2. **Grad-CAM** generates a saliency map showing which regions of the image
   drove that prediction.
3. The four **ABCD dermatoscopic criteria** (Asymmetry, Border, Color,
   differential Structures) are computed *directly from the Grad-CAM map* —
   not from raw pixels — ensuring every feature reflects what the network
   actually attended to.
4. A **Mamdani Fuzzy Inference System** combines the four ABCD scores with
   `P(malignant)` through 16 clinically-grounded IF-THEN rules, producing a
   risk score in `[0, 10]` with a natural-language explanation for each lesion.

```
Dermoscopic image
       │
       ▼
EfficientNet-B0 (fine-tuned) ──────────► P(malignant)
       │
       ▼
Grad-CAM saliency map
       │
       ▼
GrabCut segmentation ──► ABCD feature extraction
   A = geometric asymmetry of the segmented mask
   B = border-saliency concentration (quantile-normalized)
   C = color-zone saliency dispersion (K-Means in CIE L*a*b*)
   D = LBP entropy of the saliency map
       │
       ▼
Mamdani Fuzzy Inference System (5 inputs: A, B, C, D, P)
   16 IF-THEN rules, 3 hierarchical layers
       │
       ▼
Risk score [0, 10] + linguistic level + natural-language explanation
```

## Results

Evaluated on a held-out test set (`n=600`) from the HAM10000 dataset:

| | EfficientNet-B0 (network only) | Mamdani Fuzzy Layer |
|---|---|---|
| AUC-ROC | 0.927 | 0.874 |
| Sensitivity | 88.1% | 74.5% |
| Specificity | 82.6% | 81.8% |
| F1-score | — | 0.784 |

5-fold stratified cross-validation on the training set (`n=2,400`):

| | Network | Fuzzy Layer |
|---|---|---|
| AUC-ROC | 0.932 ± 0.009 | 0.879 ± 0.011 |
| Sensitivity | — | 0.678 ± 0.051 |
| Specificity | — | 0.902 ± 0.062 |
| F1-score | — | 0.769 ± 0.016 |

The gap between network and fuzzy AUC is the **interpretability overhead**:
the performance cost of replacing a direct threshold on `P` with a fully
explainable linguistic reasoning pipeline. See [Limitations](#limitations--future-work)
for an honest discussion of the sensitivity/specificity trade-off.

## Repository Structure

```
.
├── notebook/
│   └── skin_cancer_gradcam_fuzzy.ipynb   # full pipeline, documented
├── outputs/
│   ├── relatorio_fuzzy_resultados.xlsx   # per-lesion results (3 sheets)
│   ├── fig2_membership_functions.png     # calibrated MFs (paper Fig. 2)
│   └── fig3_representative_cases.png     # case studies (paper Fig. 3)
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**Dependencies:**
```
torch, torchvision
opencv-python
scikit-image
scikit-learn
scikit-fuzzy
pandas, numpy
matplotlib
openpyxl
tqdm
```

## Dataset

[HAM10000 — Human Against Machine with 10,000 training images](https://doi.org/10.1038/sdata.2018.161)
(Tschandl et al., 2018), available on
[Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

The seven original diagnostic categories are binarized into **malignant**
(melanoma, basal cell carcinoma, actinic keratosis) and **benign** (all
remaining categories).

## Usage

1. Download the HAM10000 dataset and update the paths in the configuration
   cell of the notebook (`METADATA_CSV`, `IMG_DIRS`).
2. Run the notebook top to bottom. It will:
   - Fine-tune (or load a checkpoint of) EfficientNet-B0
   - Calibrate the fuzzy membership functions on the training set
   - Run 5-fold cross-validation
   - Evaluate on the held-out test set
   - Export a clinician-friendly `.xlsx` report and the paper's figures

A pretrained checkpoint can be supplied via `MODEL_PT` to skip fine-tuning
and go directly to Grad-CAM extraction and fuzzy inference.

## Output Report

The pipeline generates an `.xlsx` file with three sheets, designed for direct
use by clinical teams without programming expertise:

- **Results** — one row per lesion: all ABCD/P scores, fuzzy risk score,
  activated rules, and a natural-language explanation.
- **Metrics** — summary comparison across baseline, network-only, and fuzzy
  configurations.
- **Fuzzy Rules** — the complete 16-rule base with clinical rationale for
  each rule.

## Limitations & Future Work

- **Specificity (81.8%)** still flags a non-trivial number of benign lesions
  as suspicious.
- The rule base's reliance on **joint `P`–`A` confirmation** in its
  highest-risk tier lowers sensitivity for confident network predictions that
  lack strong geometric asymmetry — identified as the main driver of false
  negatives in this version. Revisiting this weighting is a priority for the
  next iteration.
- Experiments used a 3,000-image subset, selected sequentially
  (`.head(N)`) rather than via random sampling — this does not guarantee the
  class balance matches the full dataset's prevalence. A larger,
  randomly-sampled, class-balanced subset is recommended for future work.
- The framework has not yet been evaluated in a prospective clinical setting.

## Citation

If you use this code or framework, please cite:

```bibtex
@inproceedings{buttow2025interpretable,
  title     = {A Multi-Level Interpretable Framework for Skin Cancer
               Classification Using Deep Learning and Fuzzy Systems},
  author    = {B{\"u}ttow, Matheus N. and Santos, Helida S. and
               Rodrigues, Rafaella M. and Dimuro, Gra{\c{c}}aliz and
               Lucca, Giancarlo and Asmus, Tiago},
  booktitle = {Brazilian Conference on Intelligent Systems (BRACIS)},
  year      = {2025}
}
```

## License

[MIT](LICENSE) — or your preferred license; update before publishing.

## Acknowledgments

Developed at the Federal University of Rio Grande (FURG), Rio Grande, RS,
Brazil. Parts of this manuscript and codebase were drafted with the
assistance of the AI system Claude; all content was critically reviewed and
validated by the authors, who take full responsibility for its accuracy.
