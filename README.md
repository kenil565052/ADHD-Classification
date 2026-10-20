# ADHD Detection Using EEG Signals

An interpretable machine learning pipeline for classifying EEG recordings into **ADHD** and **Control** groups using handcrafted signal features and subject-wise cross-validation.

## Overview

Attention-Deficit/Hyperactivity Disorder (ADHD) is a neurodevelopmental condition. EEG signals provide a way to investigate differences in brain activity between individuals with ADHD and control participants.

This project develops an EEG-based classification pipeline that combines statistical, frequency-domain, nonlinear, wavelet, and connectivity features with machine learning. A key focus is evaluating performance at the **subject level**, rather than treating multiple EEG segments from the same participant as independent samples.

## Dataset

- **Source:** [EEG Dataset for ADHD — Kaggle](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd)
- **Total participants:** 121
- **ADHD group:** 61 participants
- **Control group:** 60 participants
- **Sampling frequency used:** 128 Hz
- **Selected EEG channels:** 8

The selected channels cover three brain regions:

| Region | EEG channels |
|---|---|
| Frontal | F3, F4, Fz |
| Parietal | P3, P4, Pz |
| Occipital | O1, O2 |

## Research Motivation

Many EEG classification experiments divide data at the segment level. When segments from the same participant occur in both training and testing sets, the evaluation can be influenced by participant-specific patterns, potentially overstating generalization to unseen individuals.

This project addresses that concern by grouping EEG windows by participant during cross-validation. It also explores a broader feature set than a single EEG band or feature, combining signal statistics, spectral characteristics, nonlinear complexity, wavelet energies, and selected cross-channel features.

**Main objectives**
- Evaluate classification using participant-separated folds.
- Explore EEG features from frontal, parietal, and occipital regions.
- Combine multiple signal-processing feature families.
- Use feature selection and a machine learning pipeline.
- Examine prediction probabilities, confusion matrices, and misclassified participants.

## Methodology

### 1. EEG Windowing

Each participant's EEG data is divided into non-overlapping windows of **4 seconds**. At 128 Hz, each window contains 512 samples per channel.

Participant IDs are retained as group labels so windows from the same participant can be kept together during cross-validation.

### 2. Feature Extraction

The pipeline extracts multiple feature types from each selected channel.

| Feature family | Extracted features |
|---|---|
| Statistical | Mean, standard deviation, variance, skewness, kurtosis |
| Frequency-domain | Theta, alpha, and beta band power |
| Band-power ratio | Theta-to-beta ratio |
| Nonlinear | Higuchi Fractal Dimension, spectral entropy, sample entropy |
| Wavelet | Daubechies-4 wavelet coefficient energies across four decomposition levels |
| Connectivity | Weighted Phase Lag Index (wPLI) between Fz and Pz |
| Derived features | Frontal alpha asymmetry, complexity-to-theta-power ratio, and an Fz/Pz feature interaction |

Band power is estimated using Welch's power spectral density method. These features are combined into a feature vector for each EEG window.

### 3. Machine Learning Pipeline

The current implementation uses a scikit-learn pipeline consisting of:

1. **StandardScaler** — standardizes the extracted features.
2. **SelectKBest with ANOVA F-test** — selects features relevant to the classification task.
3. **Support Vector Classifier (SVC)** — performs ADHD versus Control classification using an RBF kernel, class balancing, and probability estimates.

`GridSearchCV` explores different feature-selection sizes and SVM regularization values to identify a suitable configuration.

### 4. Subject-Wise Cross-Validation

The notebook uses **five-fold GroupKFold cross-validation**, grouping windows by participant ID.

Predictions for individual windows are aggregated by participant using the mean predicted ADHD probability. A probability threshold of 0.5 is then used to assign the final subject-level class.

This is intended to assess performance on participants excluded from the corresponding training fold.

### 5. Evaluation and Visualization

The notebook includes:
- Subject-level accuracy, precision, recall, and F1-score.
- A confusion matrix for ADHD and Control predictions.
- Accuracy comparisons across the five folds.
- Prediction-probability distributions.
- Inspection of misclassified participants and false-positive/false-negative errors.

The selected pipeline is also saved as a `.pkl` file using `joblib` for subsequent use.

## Results

The current notebook reports the following cross-validation results:

| Metric | Result |
|---|---:|
| Subject-level accuracy | 82.64% |
| ADHD precision | 0.82 |
| ADHD recall | 0.84 |
| ADHD F1-score | 0.83 |
| Misclassified participants | 21 of 121 |
| Mean fold accuracy | 82.63% |
| Standard deviation across folds | 3.16 percentage points |

The mean fold accuracy and standard deviation are calculated across five folds. The subject-level classification report is based on the aggregated cross-validation predictions.

These results are specific to this dataset and evaluation setup. They should not be interpreted as evidence of clinical diagnostic performance or generalization to independent populations.

## Technologies Used

- Python
- NumPy and Pandas
- SciPy
- AntroPy
- PyWavelets
- scikit-learn
- Matplotlib and Seaborn
- Joblib

## Getting Started

### Requirements

Install the required Python packages:

```bash
pip install numpy pandas scipy antropy PyWavelets tqdm matplotlib seaborn scikit-learn joblib
```

### Dataset Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd).
2. Ensure the `adhdata.csv` file is available in your environment.
3. Update the CSV path in the notebook to match your local or Kaggle dataset location.
4. Run the notebook cells in sequence to extract features, train the model, evaluate predictions, and save the trained pipeline.

The notebook currently uses a Kaggle-specific input path, which must be adjusted when running outside Kaggle.

## Limitations and Future Work

- Evaluate the final model on a genuinely independent, participant-separated test set.
- Improve validation of feature selection and hyperparameter tuning within the cross-validation procedure.
- Compare SVM performance with Random Forest, KNN, and Gaussian Process classifiers.
- Expand systematic comparisons of regional features and inter-regional ratios.
- Investigate feature importance and explainability to make predictions easier to interpret.
- Validate the approach on additional datasets and populations before considering clinical applications.

**Important:** This is a research and educational project, not a standalone diagnostic tool. EEG-based predictions must not replace assessment by qualified healthcare professionals.

## References

1. [EEG Dataset for ADHD — Kaggle](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd)
2. [ADHD Detection from EEG Using Optimal Channel and Feature Selection — IEEE Xplore](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10091501)
3. [ADHD AID Research Paper — MDPI](https://www.mdpi.com/2313-7673/9/3/188)
4. [Investigating EEG as a Tool for ADHD Diagnosis — ResearchGate](https://www.researchgate.net/publication/385812820_Investigating_EEG_as_a_Tool_for_ADHD_Diagnosis)
5. [Automatic Identification of Children with ADHD from EEG Brain Waves — MDPI](https://www.mdpi.com/2624-6120/4/1/10)
6. [EEG Data for ADHD / Control Children — IEEE DataPort](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)
