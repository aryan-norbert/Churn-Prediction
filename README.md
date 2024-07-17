# Telco Customer Churn Prediction

This project aims to predict customer churn for a telecommunications company using machine learning models. The dataset is sourced from Kaggle and includes customer details such as tenure, service usage, and charges. The project involves data preprocessing, model training, evaluation, and visualization of key insights.

## Project Structure

- `data/`: Contains the dataset.
- `notebooks/`: Jupyter Notebook for data analysis and model training.
- `src/`: Python scripts for data preprocessing, model training, evaluation, and visualization.
- `models/`: Directory to store the trained models.
- `requirements.txt`: List of dependencies.
- `README.md`: Project overview and instructions.

## Setup Instructions

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/telco-churn-prediction.git
    cd telco-churn-prediction
    ```

2. **Create a virtual environment and activate it:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Download the dataset using Kaggle API:**

    ```sh
    kaggle datasets download -d blastchar/telco-customer-churn
    ```

5. **Unzip the downloaded file:**

    ```python
    import zipfile
    with zipfile.ZipFile("telco-customer-churn.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    ```

6. **Run the data preprocessing and model training script:**

    ```sh
    python src/preprocess_and_train.py
    ```

## Key Features

1. **Data Preprocessing**:
    - Downloading and unzipping the dataset.
    - Handling missing values and encoding categorical variables.
    - Scaling numerical features.

2. **Model Training**:
    - Splitting the dataset into training and testing sets.
    - Training a Random Forest Classifier.
    - Evaluating the model using classification reports and ROC AUC score.

3. **Visualization and Insights**:
    - Plotting feature importances.
    - Creating a correlation matrix to understand feature relationships.

## Usage

1. **Run the model training script** to preprocess data, train the model, and evaluate its performance:

    ```sh
    python src/preprocess_and_train.py
    ```

2. **Visualize the results**:
    - Feature importances bar plot.
    - Correlation matrix heatmap.

## License

This project is licensed under the MIT License.
