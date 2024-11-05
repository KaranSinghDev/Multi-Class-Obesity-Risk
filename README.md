Problem:
Obesity is a leading contributor to cardiovascular diseases and other chronic health issues worldwide. Predicting an individual’s risk of obesity based on personal and lifestyle factors is crucial for targeted interventions, yet current models often lack accuracy or require complex inputs. An accessible, high-accuracy tool for predicting obesity risk can bridge this gap and provide actionable insights.

Solution:
This project applies an XGBoost classifier to predict obesity risk with high accuracy, using input features such as age, gender, dietary habits, and physical activity. Unlike simpler models, this approach captures complex interactions between features, offering a more robust risk assessment than typical single-variable models.

Data:
The dataset for this project was derived from a deep learning model trained on the original Obesity and Cardiovascular Disease (CVD) Risk dataset. It contains features on individual characteristics and habits (e.g., age, weight, exercise frequency), with both training and test sets available.
Source: Kaggle Playground Series S4E2

Model:
An XGBoost Classifier was used for this task, with key hyperparameters:

Objective: multi:softmax (for multi-class classification)
n_estimators: 100
learning_rate: 0.15
Device: cuda:0 (for GPU acceleration)
eval_metric: merror
Evaluation
The model is evaluated using accuracy and cross-validation (20-fold CV) to ensure reliability and generalization.
Cross-Validation Mean Accuracy: 0.907

Citation
Walter Reade and Ashley Chow. Multi-Class Prediction of Obesity Risk. https://kaggle.com/competitions/playground-series-s4e2, 2024. Kaggle.

