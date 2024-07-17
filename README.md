# Telco Customer Churn Prediction Project

This project focuses on predicting customer churn in a telecommunications company using machine learning techniques. The dataset used is the Telco Customer Churn dataset available on Kaggle.

### Methodology:
1. **Data Acquisition**: The dataset is downloaded directly from Kaggle using the Kaggle API and then extracted using Python's `zipfile` module. The dataset contains information about customer demographics, services subscribed, and churn status.

2. **Data Cleaning and Preprocessing**:
   - **Handling Missing Values**: The `TotalCharges` column is converted to numeric values and missing values are imputed with the median.
   - **Encoding Categorical Variables**: Categorical variables are encoded using one-hot encoding to convert them into a format suitable for machine learning models.
   - **Feature Selection**: Irrelevant columns, such as `customerID`, are dropped from the dataset to focus on relevant features for prediction.

3. **Data Splitting and Scaling**:
   - The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection`.
   - Numeric features are standardized using `StandardScaler` to ensure all features are on the same scale, which is crucial for models like RandomForestClassifier.

4. **Model Building and Evaluation**:
   - **Random Forest Classifier**: This supervised learning algorithm is chosen for its ability to handle complex datasets and capture feature interactions effectively.
   - The model is trained on the training data (`X_train` and `y_train`) and evaluated on the test data (`X_test` and `y_test`).
   - Evaluation metrics include `classification_report` for precision, recall, F1-score, and `roc_auc_score` for the ROC AUC score, which measures the model's ability to distinguish between churn and non-churn customers.

5. **Visualization and Insights**:
   - **Feature Importance**: The project visualizes the top 10 most important features using a horizontal bar plot. This helps in understanding which features contribute the most to predicting customer churn.
   - **Correlation Matrix**: A heatmap of the correlation matrix is plotted using `seaborn` and `matplotlib`, providing insights into the relationships between different variables in the dataset.

### Libraries Used:
- Python 3.x
- Pandas: Data manipulation and preprocessing.
- NumPy: Numerical operations and array manipulation.
- Scikit-learn: Machine learning algorithms, including RandomForestClassifier for prediction and model evaluation.
- Matplotlib: Plotting feature importances and correlation matrix visualization.
- Seaborn: Enhanced heatmap visualization for correlation matrix.

