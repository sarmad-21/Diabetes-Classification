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
- TPR: 0.2189842976375725
- FPR: 0.038289784047450016
- TNR: 0.96171021595255
- FNR: 0.7810157023624275
- Probability of Misclassification: 0.1417730999684642
- Cross Validation Accuracy: 

Metrics for LDA PCA Data:
- TPR: 0.07738011034092517
- FPR: 0.018892985549728628
- TNR: 0.9811070144502714
- FNR: 0.9226198896590748
- Probability of Misclassification: 0.14480842005676442
- Cross Validation Accuracy: 

Metrics for QDA Original Data:
- TPR: 0.599519026736455
- FPR: 0.2189067259028557
- TNR: 0.7810932740971442
- FNR: 0.40048097326354504
- Probability of Misclassification: 0.24420529801324503
- Cross Validation Accuracy: 

Metrics for QDA PCA Data:
- TPR: 0.2315744801244872
- FPR: 0.08823596766436897
- TNR: 0.9117640323356311
- FNR: 0.7684255198755128
- Probability of Misclassification: 0.1830061494796594
- Cross Validation Accuracy: 
