# melanoma-detection-ml
# Melanoma Detection with Deep Learning

Deep learning project for classifying dermoscopic skin lesion images from the HAM10000 dataset.  
The repository contains Google Colab notebooks for training and evaluating an EfficientNetB0-based model.

> **Medical disclaimer:** This project is for educational and research purposes only. It is not a medical device and must not be used for clinical diagnosis. Skin lesions should always be assessed by a qualified dermatologist.

## Project overview

The goal of the project is to detect melanoma from dermoscopic images using convolutional neural networks and transfer learning.

The project includes two approaches:

1. **Multiclass classification** of seven HAM10000 lesion categories:
   - `akiec` — actinic keratoses and intraepithelial carcinoma
   - `bcc` — basal cell carcinoma
   - `bkl` — benign keratosis-like lesions
   - `df` — dermatofibroma
   - `mel` — melanoma
   - `nv` — melanocytic nevi
   - `vasc` — vascular lesions

2. **Binary classification**:
   - `mel` — melanoma
   - `non_mel` — all other lesion types

## Dataset

Dataset used: **HAM10000 — Human Against Machine with 10000 training images**  
Source: Kaggle dataset `kmader/skin-cancer-mnist-ham10000`

The dataset is not included in this repository because of size and licensing considerations. Download it directly from Kaggle before running the notebooks.

## Model

The model uses transfer learning with:

- EfficientNetB0 pretrained on ImageNet
- image resizing to `224 x 224`
- data augmentation
- dropout and batch normalization
- class weighting for imbalanced classes
- fine-tuning of selected EfficientNet layers

## Results

### Multiclass model

Best tested multiclass model reached approximately:

| Metric | Value |
|---|---:|
| Accuracy | 0.79 |
| Weighted F1-score | 0.79 |
| Melanoma precision | 0.49 |
| Melanoma recall | 0.46 |
| Melanoma F1-score | 0.47 |

### Binary melanoma model

At threshold `0.50`:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| non_mel | 0.94 | 0.88 | 0.91 |
| mel | 0.35 | 0.53 | 0.42 |

Overall accuracy: **0.84**

At threshold `0.30`, melanoma recall increased:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| non_mel | 0.96 | 0.74 | 0.84 |
| mel | 0.26 | 0.74 | 0.39 |

This shows the clinical trade-off between sensitivity and precision: lowering the decision threshold detects more melanoma cases but increases false positives.

## Repository structure

```text
melanoma-detection-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_training_efficientnet_ham10000.ipynb
│   └── 02_binary_melanoma_evaluation.ipynb
│
├── src/
│   └── predict.py
│
├── reports/
│   └── figures/
│
└── models/
    └── .gitkeep
```

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/melanoma-detection-ml.git
cd melanoma-detection-ml
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Configure Kaggle credentials

Do **not** hardcode your Kaggle API key in the notebook.

Recommended options:

- upload `kaggle.json` manually in Google Colab, or
- use Colab Secrets, or
- set environment variables locally.

### 4. Run notebooks

Open the notebooks in Google Colab and run them in order:

1. `notebooks/01_training_efficientnet_ham10000.ipynb`
2. `notebooks/02_binary_melanoma_evaluation.ipynb`

## Future improvements

- Add ROC curve and AUC plots
- Add Grad-CAM visualizations for model explainability
- Compare EfficientNetB0 with ResNet50 and MobileNetV2
- Add a small Streamlit demo app
- Improve melanoma recall using threshold tuning and stronger augmentation
- Use cross-validation for more reliable evaluation

## Important notes

- The model was trained on a public dataset and may not generalize to real clinical conditions.
- Dataset imbalance strongly affects melanoma detection.
- High accuracy alone is not sufficient for medical screening tasks; recall, precision, confusion matrix, ROC-AUC and calibration should also be analyzed.
