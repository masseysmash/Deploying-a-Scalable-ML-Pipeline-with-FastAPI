# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Developed by Andrew Massey for Udacity. 
4.12.2024. 
Model Verison 1.0.0. 
Logistic Regression Model

## Intended Use
Intended use is to predict salary levels based on census data. Intended users are non-profits looking for donors.

## Training Data
Training Data is Census data. Categorical columns are processed using One Hot Encoding. Binary columns are processed using BinaryLabelizer

## Evaluation Data
Evaluation was 20% of the base Census Dataset. 

## Metrics

Precision - 0.7150 
Recall - 0.2683
F1 Score - 0.3902

## Ethical Considerations
No PI information is included in this dataset. None of the data is sensitive or has any inherent risk.

## Caveats and Recommendations
Checks for bias are reccomended for future development. 
