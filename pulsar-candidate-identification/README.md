# Pulsar Candidate Identification — Real HTRU2 Survey Data
 
## Data disclosure — this project uses REAL data
 
This project is built on **genuinely real astronomical survey data**: 
the HTRU2 dataset — 17,898 real pulsar candidates from the actual High Time Resolution Universe Survey (South), conducted with the Parkes radio telescope.
 
The dataset's own creator,**Dr. Rob Lyon** 
(University of Manchester, Jodrell Bank Centre for Astrophysics), hosts the original
data in his public repository (`scienceguyrob/PulsarFeatureLab`), which was fetched here via a
repository archive download. This is the same authoritative primary-source data distributed through the
UCI/Kaggle mirrors, verified against documented dataset statistics (17,898 total; 1,639 pulsars; 16,259
non-pulsars — exact match).
 
**Citation:** Lyon, R. J., Stappers, B. W., Cooper, S., Brooke, J. M., Knowles, J. D. (2016). "Fifty
Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification
approach." *Monthly Notices of the Royal Astronomical Society*, 459(1), 1104-1123.
DOI: 10.1093/mnras/stw656. Dataset DOI: 10.6084/m9.figshare.3080389.v1
 
## What this project demonstrates
 
- Real-world imbalanced classification methodology (9.9:1 class ratio) — accuracy is explicitly the
  wrong metric here, and the project shows why
- Feature engineering validation against real published results (pulse-profile kurtosis dominance)
- A genuine methodological comparison (8 vs. 30 features) rather than an assumed "more is better"
- Honest handling of a real, severe class imbalance rather than a synthetically balanced dataset

## Pipeline
 
```
reference_data.py           Full provenance documentation, feature definitions, verified counts
load_and_validate.py        Loads + validates real data against documented statistics
train_classifiers.py        Logistic Regression / Random Forest / Gradient Boosting, imbalance-aware
compare_feature_sets.py     8-feature vs. 30-feature (Lyon + Thornton) comparison
make_figures.py             All plots
notebooks/pulsar_identification.ipynb   Full executed walkthrough
```
 
## Headline results
 
| Model | ROC-AUC | PR-AUC | F1 (pulsar) |
|---|---|---|---|
| Logistic Regression | 0.975 | 0.934 | 0.844 |
| Random Forest | 0.978 | 0.932 | 0.878 |
| **Gradient Boosting** | 0.977 | 0.930 | **0.887** |
 
| Feature set | Features | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|---|
| Lyon 2015 (standard) | 8 | 0.978 | 0.932 | 0.878 |
| **Lyon + Thornton (combined)** | 30 | **0.987** | **0.954** | **0.901** |
 
**Top feature (Random Forest importance): excess kurtosis of the integrated pulse profile (34.9%)** —
matching the real published finding in Lyon et al. (2016) that pulse-profile shape statistics, not
DM-SNR curve statistics, carry the most discriminating signal for this task.
 
1. With a 9.9:1 class imbalance, a trivial "always non-pulsar" classifier already scores ~91% accuracy
   — accuracy alone is meaningless here, and using ROC-AUC/PR-AUC/per-class precision-recall instead is
   the correct methodology, not an optional refinement.
2. Feature importance here isn't just an artifact of this run — it independently reproduces the real
   published result (pulse-profile kurtosis dominance) from Lyon et al. 2016, which is a strong
   validation signal.
3. Gradient Boosting's higher precision / lower recall vs. Random Forest's higher recall / lower
   precision is a real trade-off with a genuine operational interpretation: survey follow-up time
   constraints favor precision, discovery-focused surveys favor recall.
4. The 8-vs-30-feature comparison answers a real methodological question with data rather than
   assumption — richer feature engineering does help here, measurably, though the original 8 features
   already capture most of the signal.

## Honest limitations
 
- Only the tabular Lyon/Thornton feature sets are used here, not the raw folded-profile or DM-SNR-curve
  time series themselves (which would enable a CNN-based comparison, as in Project 4's galaxy
  morphology project) — a natural extension.
- Class weighting (rather than resampling techniques like SMOTE) was used to handle imbalance; a fuller
  study would compare multiple resampling strategies.
- No cross-validation was performed beyond a single train/test split; a production-grade evaluation
  would use stratified k-fold CV to quantify metric variance.