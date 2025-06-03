# Detecting Diabetic/Prediabetic Individuals 

## Overview 
This project focuses on classifying individuals as diabetic/prediabetic or non-diabetic using health, lifestyle, and demographic related features. We performed data preprocessing, exploratory data analysis, and basic feature extraction to prepare the dataset for modeling. Two classical statistical models are then used to predict whether an individual is diabetic/prediabetic or non-diabetic: **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)**. This projects objective is to apply machine learning techniques to public health data and assess their effectiveness in identifying at risk individuals.

## Dataset 
The dataset used in this project is the CDC Diabetes Health Indicators dataset from the UC Irvine Machine Learning Repository. 
- Samples: 253,680 individuals
- Target Variable: Diabetes_binary (1.0: Diabetic/Pre-Diabetic, 0.0: Non-diabetic)
- 21 health, lifestyle, and demographic related features

Link to dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

## Libraries 
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn

## Models 
- LDA: A plug in MAP (Maximum A Posteriori) classifier where likelihood functions for each class are assumed to be Gaussian with different means but the same covariance matrix. LDA estimates the mean vectors for each class, the shared covariance matrix across classes, and the prior probabilities for each class using the training data. These estimates are then used to compute discriminant functions for each class. A data point is then assigned to the class with the highest discriminant score which is proportional to the log of the posterior probability.  

<pre>
ŷ<sub>LDA</sub> = arg max<sub>j</sub>[x<sup>T</sup>Ĉ<sup>−1</sup>μ̂<sub>j</sub> − 1/2μ̂<sub>j</sub><sup>T</sup>Ĉ<sup>−1</sup>μ̂<sub>j</sub> + ln(π̂<sub>j</sub>)]
</pre>

- QDA: A plug in MAP (Maximum A Posteriori) classifier where likelihood functions for each class are assumed to be Gaussian with different means and different covariance matrices. QDA estimates the mean vectors for each class, the covariance matrices for each class, and the prior probabilities for each class using the training data. These estimates are then used to compute discriminant functions for each class. A data point is then assigned to the class with the highest discriminant score which is proportional to the log of the posterior probability.  

<pre>
ŷ<sub>QDA</sub> = arg max<sub>j</sub>[−1/2x<sup>T</sup>Ĉ<sub>j</sub><sup>−1</sup>x + x<sup>T</sup>Ĉ<sub>j</sub><sup>−1</sup>μ̂<sub>j</sub> − 1/2μ̂<sub>j</sub><sup>T</sup>Ĉ<sub>j</sub><sup>−1</sup>μ̂<sub>j</sub> − 1/2ln(det(Ĉ<sub>j</sub>)) + ln(π̂<sub>j</sub>)]
</pre>

## Performance and Evaluation 
The models were evaluated using cross validation accuracy, confusion matrices, and misclassification probabilities. Additionally, performance was compared between models trained on the original dataset and those trained on the dataset with PCA applied to it. When applying PCA we retained enough components to preserve 90% of the variance in the data. Performance of both LDA and QDA trained on the original dataset and the dataset with PCA applied to it are shown below. 

Metrics for LDA Original Data:
- Cross Validation Accuracy: 0.8607
- TPR: 0.2190
- FPR: 0.0383
- TNR: 0.9617
- FNR: 0.7810
- Probability of Misclassification: 0.1418

Metrics for LDA PCA Data:
- Cross Validation Accuracy: 0.8568
- TPR: 0.0774
- FPR: 0.0189
- TNR: 0.9811
- FNR: 0.9226
- Probability of Misclassification: 0.1448

Metrics for QDA Original Data:
- Cross Validation Accuracy: 0.7558
- TPR: 0.5995
- FPR: 0.2189
- TNR: 0.7811
- FNR: 0.4005
- Probability of Misclassification: 0.2442

Metrics for QDA PCA Data:
- Cross Validation Accuracy: 0.8160
- TPR: 0.2316
- FPR: 0.0882
- TNR: 0.9118
- FNR: 0.7684
- Probability of Misclassification: 0.1830
