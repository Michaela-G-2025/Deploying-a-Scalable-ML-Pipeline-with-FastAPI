# Model Card

For additional information, see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Logistic Regression Classifier. It is trained to predict if a person's income is greater or less than $50,000 per year based on demographic information and employment. 
* Version 1.0
* Developer: Michaela/Udacity Student
* Date: 7/12/25

## Intended Use
This model should be used for binary classification of a person's income based on data from the US Census. It could be used for demographic analysis or target marketing.

## Training Data
This model was trained on the Adult Census Income dataset. 
* Features include: age, workclass, education, marital-status, etc.
* The target: predicting salary (<=$50k and >$50k)
* The training set is 80% of the overall dataset, which is 32,561

## Evaluation Data
This model was evaluated using 20% of the data from the Adult Census Income dataset.
* Size: The test dataset contains 20% of the original dataset, which is 32,561.

## Metrics
The model was evaluated using Precision, Recall, and the F1 Score. 
* Overall performance:                                               
  * Precision: 0.70
  * Recall: 0.27
  * F1 Score: 0.39
* Performance on Categorical slices: The performance varies across categories. The precision is reasonable, but the recall remains low.
  * Workclass F1 Score range: 0.31-0.50
  * Education F1 score range: 0.00 (for lower education classes)- 0.61 (for higher education classes)
  * Martial Status F1 score range: 0.00-0.48
  * Occupation F1 score range: 0.09-0.49
  * Relationship F1 score range: 0.24-0.40
  * Race F1 score range: 0.29-0.48
  * Sex F1 score: M- 0.40 F- 0.39
  * Native Country F1 score range: 0.00-0.40 (some values are higher because of having minimal samples)

## Ethical Considerations
The dataset is known to contain demographic information that could be sensitive. 
* Bias could include race, sex, education, and native country.
* Data may not accurately reflect the current demographic. 

## Caveats and Recommendations
This model was trained on a limited dataset and may not generalize well. 
This model also has an overall low recall score, indicating that it defines a significant portion of the population. 

                                                                                                   
