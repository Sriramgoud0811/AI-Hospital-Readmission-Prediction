
# AI Hospital Readmission Prediction

An AI-powered healthcare machine learning application that predicts hospital length of stay and 30-day patient readmission risk using machine learning models and a FastAPI backend.

## Project Overview

This project uses hospital patient and encounter data to develop two machine learning solutions:

1. **Regression Model:** Predicts the expected duration of a patient's hospital stay.
2. **Classification Model:** Predicts whether a patient may be readmitted to the hospital within 30 days.

The trained models are integrated into a REST API using FastAPI. The application is designed for deployment using platforms such as Render.

> This project is intended for educational and analytical purposes. Predictions should not be used as a replacement for professional medical judgment.

---

## Business Problem

Hospitals need to manage patient resources, improve discharge planning, and identify patients who may require additional follow-up care.

This project addresses two business problems:

### Problem 1: Hospital Length of Stay Prediction

Predict the number of days a patient may stay in the hospital.

**Target variable:**

`time_in_hospital`

The prediction supports:

- Hospital resource planning
- Bed availability management
- Discharge planning
- Operational analysis

### Problem 2: 30-Day Readmission Prediction

Predict whether a patient is likely to be readmitted to the hospital within 30 days.

**Target variable:**

`readmission risk`

Binary classification:

- `0` = No readmission within 30 days
- `1` = Readmission within 30 days

The prediction can support:

- Patient risk identification
- Follow-up planning
- Healthcare resource management
- Preventive care analysis

---

## Project Objectives

- Collect and understand the hospital readmission dataset
- Perform data cleaning and preprocessing
- Conduct exploratory data analysis
- Identify important features and patterns
- Build regression and classification models
- Evaluate model performance
- Save trained machine learning models
- Develop REST API endpoints using FastAPI
- Prepare the project for deployment

---

## Dataset

The project uses hospital patient and encounter records.

The dataset contains information related to:

- Patient demographics
- Patient medical history
- Hospital encounters
- Admission details
- Discharge information
- Medical procedures
- Medication-related information
- Hospital stay duration
- Readmission status

The dataset was inspected, cleaned, and prepared before model training.

The detailed dataset information and column descriptions are available in the `reports` directory.

---

## Machine Learning Workflow

The project follows the complete machine learning lifecycle:

1. Dataset collection
2. Dataset understanding
3. Data validation
4. Data cleaning
5. Exploratory Data Analysis
6. Feature engineering
7. Train-test splitting
8. Data preprocessing
9. Model training
10. Model evaluation
11. Model saving
12. FastAPI integration
13. API testing
14. Deployment preparation

---

## Project Structure

```text
AI-Hospital-Readmission/
│
├── app/
│   ├── database/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   └── main.py
│
├── data/
│   └── raw/
│
├── notebook/
│   ├── Classification_Model_work.ipynb
│   ├── Regression_Model_work.ipynb
│   ├── datavalidation and eda.ipynb
│   └── app/
│       └── model/
│           ├── hospital_readmission_xgboost_model.pkl
│           └── hospital_readmission_xgboost_model.pkl
│
├── reports/
│   ├── AI_Hospital_Readmission_Industry_Domain_Data_Collection_Styled.docx
│   ├── Hospital_Readmission_Dataset_Column_Data_Dictionary.docx
│   └── Recommended_ML_Problems.docx
│
├── sql/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Data Collection and Understanding

The dataset was collected and examined to understand the healthcare business problem.

The following activities were performed:

- Reviewed the dataset structure
- Checked the number of rows and columns
- Studied data types
- Identified missing values
- Checked duplicate records
- Inspected categorical and numerical features
- Reviewed target variables
- Prepared a column data dictionary
- Identified possible machine learning problems

The data dictionary is available in:

```text
reports/Hospital_Readmission_Dataset_Column_Data_Dictionary.docx
```

---

## Data Validation and Cleaning

Data validation and cleaning were performed before model development.

Activities included:

- Missing-value analysis
- Duplicate-value checking
- Data-type validation
- Categorical-value inspection
- Numerical-column inspection
- Outlier investigation
- Target-variable validation
- Feature consistency checks
- Preprocessing preparation

The data validation notebook is available in:

```text
notebook/datavalidation and eda.ipynb
```

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand patterns in the hospital dataset.

The analysis focused on:

- Distribution of numerical variables
- Distribution of categorical variables
- Hospital stay duration
- Readmission distribution
- Patient and encounter patterns
- Feature relationships
- Potentially important predictive variables

EDA helps identify data quality problems and supports better feature selection and model development.

---

## Model 1: Regression

### Objective

Predict the expected hospital stay duration.

### Target Variable

```text
time_in_hospital
```

The target represents the number of days a patient stays in the hospital.

### Regression Workflow

- Select the regression target
- Clean and preprocess the data
- Separate features and target
- Split the dataset into training and testing data
- Train regression models
- Evaluate prediction performance
- Save the trained model
- Integrate the model with FastAPI

### Evaluation Metrics

The regression model can be evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

The detailed regression implementation is available in:

```text
notebook/Regression_Model_work.ipynb
```

---

## Model 2: Classification

### Objective

Predict whether a patient will be readmitted to the hospital within 30 days.

### Target Variable

```text
readmission risk
```

This is a binary classification problem:

| Value | Meaning |
|---|---|
| `0` | No readmission within 30 days |
| `1` | Readmission within 30 days |

### Classification Models

The project evaluates machine learning classification algorithms such as:

- Logistic Regression
- Random Forest
- Support Vector Machine
- XGBoost

### Classification Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Brier Score

Recall is important because missing a patient who may be readmitted can be significant in a healthcare risk-screening scenario.

The detailed classification implementation is available in:

```text
notebook/Classification_Model_work.ipynb
```

---

## Model Saving

Trained models are saved using the Joblib format.

Example:

```text
hospital_readmission_xgboost_model.pkl
```

The saved model can be loaded by the FastAPI application to generate predictions without retraining the model each time.

---

## FastAPI Application

FastAPI is used to create REST API endpoints for model predictions.

The API is designed to:

- Receive patient input data
- Validate incoming data
- Load the trained machine learning model
- Generate predictions
- Return prediction results in JSON format

### Technologies Used

- Python
- FastAPI
- Uvicorn
- Pydantic
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

---

## Running the Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sriramgoud0811/AI-Hospital-Readmission-Prediction.git
```

### Step 2: Open the Project Directory

```bash
cd AI-Hospital-Readmission-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment on Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Start the FastAPI Application

Use the following command if `main.py` is inside the `app` directory:

```bash
uvicorn app.main:app --reload
```

If your FastAPI file or application variable has a different name, update the command according to your project structure.

---

## API Documentation

After starting the application, open:

```text
http://127.0.0.1:8000/docs
```

FastAPI provides an interactive Swagger UI where API endpoints can be tested.

Alternative API documentation:

```text
http://127.0.0.1:8000/redoc
```

---

## API Testing

API testing should verify:

- Valid patient input
- Missing input fields
- Incorrect data types
- Invalid categorical values
- Regression prediction response
- Classification prediction response
- Error handling
- Model loading

The API should return clear JSON responses containing the prediction and relevant information.

---

## Deployment

The application is prepared for deployment using Render.

Deployment requirements include:

- A working FastAPI application
- A valid `requirements.txt` file
- Correct model file paths
- A production server command
- Proper environment configuration
- Successful API testing

Example production command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

The final deployment command must match the actual FastAPI application structure.

---

## Tools and Technologies

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Model Saving | Joblib |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Validation | Pydantic |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |
| Deployment | Render |

---

## Future Improvements

- Improve feature engineering
- Perform additional hyperparameter tuning
- Improve classification recall
- Add model explainability
- Add input validation
- Add API authentication
- Add automated testing
- Add monitoring and logging
- Build a frontend dashboard
- Deploy the application publicly

---

## Disclaimer

This application is an educational machine learning project.

The predictions are not medical diagnoses and should not be used as the sole basis for clinical decisions. Healthcare professionals must evaluate patient information before making decisions.

---

## Author

**Chinnolla Sriram Goud**

GitHub:

https://github.com/Sriramgoud0811

Project Repository:

https://github.com/Sriramgoud0811/AI-Hospital-Readmission-Prediction
```

---

## 4. Save and upload README

In VS Code:

1. Open `README.md`.
2. Paste the documentation.
3. Press **Ctrl + S**.
4. Open the terminal.
5. Run these commands one by one:

```powershell
git add README.md requirements.txt
```

```powershell
git commit -m "Add complete project documentation and requirements"
```

```powershell
git push origin main
```

### Important

Your GitHub repository will show the README automatically because it is named:

```text
README.md
```

You do **not** need to create another README through GitHub.

Also, GitHub does not display completely empty folders. If you want empty folders such as `sql`, `database`, or `schemas` to appear, add a file named:

```text
.gitkeep
```

inside each empty folder and commit it.
