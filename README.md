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
- LDA: A plug in MAP (Maximum A Posteriori) classifier where likelihood functions for each class are Gaussian with different means but the same covariance matrix. LDA estimates the mean vectors for each class, the shared covariance matrix across classes, and the prior probabilities for each class using the training data. These are then used to compute discriminant functions for each class which assigns a data point to the class with the highest posterior probability.

<pre>
ŷ<sub>LDA</sub> = arg max<sub>j</sub>[x<sup>T</sup>Ĉ<sup>−1</sup>μ̂<sub>j</sub> − 1/2 μ̂<sub>j</sub><sup>T</sup>Ĉ<sup>−1</sup>μ̂<sub>j</sub> + ln(π̂<sub>j</sub>)]
</pre>



- QDA:


## Performance and Evaluation 
The models were evaluated using 
