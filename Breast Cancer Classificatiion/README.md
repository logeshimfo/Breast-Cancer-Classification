# Breast Cancer Classification

A machine learning project for classifying breast tumors as **malignant** or **benign** using the Breast Cancer Wisconsin dataset available through `scikit-learn`.

The project covers the complete machine learning workflow, including data exploration, preprocessing, model comparison, cross-validation, hyperparameter tuning, model persistence, and deployment using Streamlit.

> **Disclaimer:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.

---

## Project Overview

Breast cancer classification is a binary classification problem where the objective is to predict whether a tumor is:

* **Malignant**
* **Benign**

The project uses numerical features describing characteristics of cell nuclei obtained from breast mass images.

The dataset is loaded directly from `scikit-learn`, so no external dataset download is required.

---

## Dataset

The project uses the `load_breast_cancer()` dataset from `scikit-learn`.

* **Samples:** 569
* **Features:** 30
* **Target classes:** 2

### Target Classes

| Label | Class     |
| ----- | --------- |
| 0     | Malignant |
| 1     | Benign    |

The 30 features describe measurements such as:

* Mean radius
* Mean texture
* Mean perimeter
* Mean area
* Mean smoothness
* Mean compactness
* Mean concavity
* Mean concave points
* Mean symmetry
* Mean fractal dimension
* Radius error
* Texture error
* Perimeter error
* Area error
* Smoothness error
* Compactness error
* Concavity error
* Concave points error
* Symmetry error
* Fractal dimension error
* Worst radius
* Worst texture
* Worst perimeter
* Worst area
* Worst smoothness
* Worst compactness
* Worst concavity
* Worst concave points
* Worst symmetry
* Worst fractal dimension

---

## Machine Learning Workflow

The project follows this workflow:

```text
Dataset
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Final Model Selection
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

---

## Models Evaluated

The following classification algorithms were explored:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Classifier (SVC)
* Decision Tree
* Random Forest
* Gradient Boosting

### Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Because this is a medical-related classification problem, **recall and F1-score** are particularly important evaluation metrics in addition to accuracy.

---

## Preprocessing

Feature scaling is important because the dataset contains features with different numerical ranges.

For models that require scaling, `StandardScaler` is used inside a `Pipeline`.

Example:

```python
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ]
)
```

Using a pipeline ensures that preprocessing and model training are performed consistently.

---

## Model Validation

A **5-fold cross-validation** strategy was used with F1-score as the scoring metric.

```python
cross_val_score(
    model,
    x_train,
    y_train,
    cv=5,
    scoring="f1"
)
```

This provides a more reliable estimate of model performance than relying only on a single train/test split.

---

## Hyperparameter Tuning

`GridSearchCV` was used to find better hyperparameters for:

* Logistic Regression
* Support Vector Classifier

The search was performed using:

* 5-fold cross-validation
* F1-score
* Parallel processing with `n_jobs=-1`

The final model selected for deployment was the **tuned Support Vector Classifier (SVC)**.

---

## Final Model

The final trained pipeline contains:

```text
StandardScaler
      ↓
SVC
```

The trained model is saved using `joblib`:

```text
models/breast_cancer_model.pkl
```

The complete preprocessing and model pipeline is stored together so that the same preprocessing can be applied during prediction.

---

## Project Structure

```text
breast-cancer-classification/
│
├── notebooks/
│   └── breast_cancer_analysis.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── breast_cancer_model.pkl
│
├── app/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Directory Description

**`notebooks/`**

Contains the exploratory analysis and experimentation performed during development.

**`src/train.py`**

Contains the reproducible training pipeline, including preprocessing, model training, cross-validation, hyperparameter tuning, evaluation, and model saving.

**`src/predict.py`**

Loads the trained model and provides prediction functionality for new input data.

**`models/`**

Contains the trained machine learning model.

**`app/app.py`**

Contains the Streamlit application used to interact with the trained model.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

Navigate into the project:

```bash
cd breast-cancer-classification
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run the training script from the project root:

```bash
python src/train.py
```

This will:

1. Load the breast cancer dataset.
2. Split the data into training and testing sets.
3. Build preprocessing pipelines.
4. Train the candidate models.
5. Perform cross-validation.
6. Perform hyperparameter tuning.
7. Evaluate the tuned models.
8. Select the final SVC model.
9. Save the trained model to:

```text
models/breast_cancer_model.pkl
```

---

## Make Predictions

The prediction module can be used to load the saved model and make predictions on new feature values.

```bash
python src/predict.py
```

The prediction pipeline expects **30 numerical features** corresponding to the dataset's feature order.

---

## Run the Streamlit Application

Start the application with:

```bash
streamlit run app/app.py
```

The Streamlit interface allows users to provide feature values and obtain a model prediction.

The application displays the predicted class and prediction probability.

---

## Example Prediction

The model produces one of two classifications:

```text
Malignant
```

or

```text
Benign
```

along with the model's predicted probability.

---

## Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Exploratory Data Analysis
* Feature/target separation
* Stratified train/test splitting
* Feature scaling
* Scikit-learn Pipelines
* Classification algorithms
* Cross-validation
* Hyperparameter tuning
* GridSearchCV
* Classification metrics
* Confusion matrix
* ROC-AUC
* Model serialization with Joblib
* Model inference
* Streamlit deployment
* Basic ML project organization

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook
* Git/GitHub

---

## Future Improvements

Possible improvements include:

* Add ROC curve visualization
* Add confusion matrix visualization to the Streamlit application
* Add feature importance/explainability analysis
* Add SHAP-based explanations
* Add automated testing
* Add logging
* Add Docker support
* Add CI/CD using GitHub Actions
* Deploy the Streamlit application publicly
* Add model versioning
* Improve the prediction interface

---

## Author

**Logeshwaran**

Computer Science | Machine Learning | Deep Learning

---

## License

This project is intended for educational and portfolio purposes.
