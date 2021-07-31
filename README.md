# leukemia-cell-classification
This repository contains a model trained on the dataset https://www.kaggle.com/andrewmvd/leukemia-classification

Training for this model was carried out by using Arkangel AI's in-house development named 'hippocrates' which trains different architectures and configurations to get the best possible performance out of a convolutional neural network for a given problem.

This repository contains the resultant model in format .h5, this model was validated under a hold out scheme by joining training and validation cohorts from the dataset and splitting this merged set into train, test and validation samples respectively. As a result, performance metrics for the test cohort are:

* sensitivity: 0.83
* specificity: 0.62
* AUC: 0.76
