# bearing-fault-detection-ML-DL-Workflows
Comparative Analysis of Machine Learning Workflows Using the CRWU Dataset

The repository cinsist of a dataset in the form of a csv file, which can be used to run this project. Also included in the repository are Python files for all the ML models, and separate files for different DL Algorithms like CNN.

The Case Western Reserve University (CWRU) dataset comprises vibration signals collected from motor bearings under diverse conditions, including variations in motor loading, bearing sizes, motor speeds, and measurement locations, such as the fan end and drive end. However, the high dimensionality and complex conditions of the CWRU dataset pose significant challenges for extracting meaningful information using conventional approaches, such as rule-based techniques and statistical methods. To address this challenge, we propose a machine learning (ML)-integrated workflow to efficiently perform fault diagnosis, achieving high classification metrics (e.g., accuracy, precision, and recall). This workflow comprises multiple stages, including data preprocessing, feature engineering, constructing multiple ML models, and evaluating model performance. By leveraging effective data preprocessing and feature engineering, we achieved high classification testing accuracies across various models, including logistic regression (93%), convolutional neural networks (CNNs) (95%), XGBoost (99%), random forest (98%), and moderate accuracy for the SVM classifier (68%). This demonstrates that, with proper data preprocessing and feature engineering, simple classification algorithms (notably, tree-based models) can achieve high performance in classification tasks using the CWRU dataset. Furthermore, a nearly perfect accuracy can be approached with appropriate hyperparameter optimization for more complex models such as CNNs. Looking forward, integrating model interpretation methods such as SHapley Additive exPlanations (SHAP) and Local interpretable model-agnostic explanations (LIME) into this workflow could further enhance this workflow by revealing feature correlations within the dataset.

