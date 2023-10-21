# Scikit-Learn Essentials: A Handy Cheatsheet for Data Scientists
This is collection of machine learning cheat sheet 
Scikit-learn (sklearn) is a popular Python library for machine learning and data analysis. It provides a wide range of functions and classes for various machine learning tasks, including classification, regression, clustering, dimensionality reduction, and more. Here's a list of some commonly used functions and classes in scikit-learn:

1. **Model Selection:**
   - `train_test_split`: Splits the dataset into training and testing sets.
   - `cross_val_score`: Performs k-fold cross-validation for model evaluation.
   - `GridSearchCV`: Conducts hyperparameter tuning using grid search.
   - `RandomizedSearchCV`: Conducts hyperparameter tuning using randomized search.

2. **Supervised Learning:**
   - `LinearRegression`: Linear regression model for regression tasks.
   - `LogisticRegression`: Logistic regression model for classification tasks.
   - `DecisionTreeClassifier` and `DecisionTreeRegressor`: Decision tree models.
   - `RandomForestClassifier` and `RandomForestRegressor`: Random forest models.
   - `SVC` and `SVR`: Support Vector Classifier and Regressor.
   - `KNeighborsClassifier` and `KNeighborsRegressor`: k-Nearest Neighbors models.
   - `GradientBoostingClassifier` and `GradientBoostingRegressor`: Gradient Boosting models.

3. **Unsupervised Learning:**
   - `KMeans`: K-Means clustering.
   - `PCA`: Principal Component Analysis for dimensionality reduction.
   - `DBSCAN`: Density-Based Spatial Clustering of Applications with Noise.
   - `IsolationForest`: Isolation Forest for anomaly detection.

4. **Preprocessing:**
   - `StandardScaler`: Standardizes features to have mean 0 and variance 1.
   - `MinMaxScaler`: Scales features to a specified range, usually [0, 1].
   - `OneHotEncoder`: Encodes categorical variables into one-hot vectors.
   - `LabelEncoder`: Encodes categorical labels into numerical values.
   - `Imputer`: Imputes missing values in datasets.

5. **Evaluation:**
   - `accuracy_score`: Computes classification accuracy.
   - `mean_squared_error`: Calculates mean squared error for regression.
   - `confusion_matrix`: Generates a confusion matrix for classification.
   - `classification_report`: Generates a classification report with various metrics.
   - `roc_curve` and `roc_auc_score`: Receiver Operating Characteristic (ROC) analysis.

6. **Feature Selection and Extraction:**
   - `SelectKBest`: Selects the top k features based on statistical tests.
   - `PCA`: Principal Component Analysis for feature extraction.
   - `RFE` (Recursive Feature Elimination): Selects features recursively based on model performance.

7. **Ensemble Methods:**
   - `VotingClassifier` and `VotingRegressor`: Combines multiple models for ensemble learning.
   - `BaggingClassifier` and `BaggingRegressor`: Bagging ensemble methods.
   - `AdaBoostClassifier` and `AdaBoostRegressor`: Adaptive Boosting ensemble methods.

8. **Neural Network Interface:**
   - `MLPClassifier` and `MLPRegressor`: Multi-layer Perceptron models.

This is not an exhaustive list, as scikit-learn offers many more functions and classes for various machine learning tasks. You can find more details and documentation on the official scikit-learn website: https://scikit-learn.org/stable/.
