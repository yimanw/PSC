# PSC
Remote sensing retrieval of phytoplankton size classes (PSC) based on machine learning in the coastal waters off the Leizhou Peninsula
This repository corresponds to the research manuscript authored by Yiman Wang et al., submitted to Ecological Informatics and currently under review.
The study focuses on the retrieval of phytoplankton size classes (PSC), including pico-, nano-, and micro-phytoplankton, in coastal waters using in situ measurements . A total of multiple field samples collected during several research cruises were used to develop and evaluate machine-learning-based retrieval models.

Machine learning models
Six machine learning algorithms were implemented and evaluated:
- Light Gradient Boosting Machine (LightGBM)
- Extreme Gradient Boosting (XGBoost)
- Random Forest (RF)
- Least Squares Support Vector Machine (LSSVM)
- Artificial Neural Network (ANN)
- Gaussian Process Regression (GPR)

What you can find

Research data:
Phytoplankton size class concentration data, The spectral variables include reflectance values and spectral indices constructed from Sentinel-2 bands (e.g., B1–B12).

Code:
MATLAB scripts for training and evaluating six machine learning models for PSC retrieval.
