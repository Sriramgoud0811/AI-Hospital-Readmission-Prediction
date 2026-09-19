from pathlib import Path
import logging
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("ai_hospital_api")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="AI Hospital Readmission & Length-of-Stay API",
    description=(
        "API for hospital length-of-stay estimation "
        "and 30-day readmission risk assessment."
    ),
    version="1.0.0"
)

# Allow the deployed frontend to call this API from a browser.
# For production, replace ["*"] with your exact frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Project Paths
# ============================================================

# Resolve the project root safely whether main.py is located in the
# repository root, an app/ folder, or another one-level subfolder.
FILE_DIR = Path(__file__).resolve().parent
CANDIDATE_APP_DIRS = [
    FILE_DIR,
    FILE_DIR.parent,
    FILE_DIR.parent.parent,
    Path.cwd(),
]

APP_DIR = next(
    (candidate for candidate in CANDIDATE_APP_DIRS
     if (candidate / "pickles").is_dir()),
    FILE_DIR.parent,
)
PICKLES_DIR = APP_DIR / "pickles"

MODEL1_DIR = PICKLES_DIR / "model1"
MODEL2_DIR = PICKLES_DIR / "model2"


# ============================================================
# Regression Artifacts
# ============================================================

REGRESSION_MODEL_PATH = (
    MODEL1_DIR / "final_xgb_regressor.pkl"
)

REGRESSION_ENCODER_PATH = (
    MODEL1_DIR / "regression_onehot_encoder.pkl"
)

REGRESSION_SCALER_PATH = (
    MODEL1_DIR / "regression_scaler.pkl"
)


# ============================================================
# Classification Artifacts
# ============================================================

CLASSIFICATION_MODEL_PATH = (
    MODEL2_DIR / "final_xgboost_classifier.pkl"
)

CLASSIFICATION_ENCODER_PATH = (
    MODEL2_DIR / "classification_onehot_encoder.pkl"
)

CLASSIFICATION_SCALER_PATH = (
    MODEL2_DIR / "classification_scaler.pkl"
)


# ============================================================
# Loaded Artifacts
# ============================================================

regression_model = None
regression_encoder = None
regression_scaler = None

classification_package = None
classification_model = None
classification_encoder = None
classification_scaler = None

classification_threshold = 0.12

artifact_errors = []


# ============================================================
# Utility: Load Pickle
# ============================================================

def load_pickle(path: Path):

    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    with path.open("rb") as file:
        return pickle.load(file)


def pydantic_to_dict(data):
    """Support both Pydantic v2 and v1 environments."""
    if hasattr(data, "model_dump"):
        return data.model_dump()
    return data.dict()


# ============================================================
# Load All Finalized Artifacts
# ============================================================

def load_artifacts():

    global regression_model
    global regression_encoder
    global regression_scaler

    global classification_package
    global classification_model
    global classification_encoder
    global classification_scaler

    global classification_threshold
    global artifact_errors

    artifact_errors.clear()

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    try:

        regression_model = load_pickle(
            REGRESSION_MODEL_PATH
        )

        regression_encoder = load_pickle(
            REGRESSION_ENCODER_PATH
        )

        regression_scaler = load_pickle(
            REGRESSION_SCALER_PATH
        )

        # Exact training structure:
        # 14 categorical inputs
        # 2354 encoded features
        # 4 scaled numerical features
        # 1 inpatient history flag
        # 2359 final model features

        if regression_encoder.n_features_in_ != 14:
            raise ValueError(
                "Regression encoder expects "
                f"{regression_encoder.n_features_in_} inputs; "
                "expected 14."
            )

        regression_encoded_output = len(
            regression_encoder.get_feature_names_out()
        )

        if regression_encoded_output != 2354:
            raise ValueError(
                "Regression encoder outputs "
                f"{regression_encoded_output} features; "
                "expected 2354."
            )

        if regression_scaler.n_features_in_ != 4:
            raise ValueError(
                "Regression scaler expects "
                f"{regression_scaler.n_features_in_} features; "
                "expected 4."
            )

        if regression_model.n_features_in_ != 2359:
            raise ValueError(
                "Regression model expects "
                f"{regression_model.n_features_in_} features; "
                "expected 2359."
            )

        logger.info(
            "Regression artifacts loaded successfully."
        )

    except Exception as exc:

        regression_model = None
        regression_encoder = None
        regression_scaler = None

        artifact_errors.append(
            f"Regression artifacts: {exc}"
        )

        logger.exception(
            "Regression artifact loading failed."
        )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    try:

        classification_package = load_pickle(
            CLASSIFICATION_MODEL_PATH
        )

        classification_encoder = load_pickle(
            CLASSIFICATION_ENCODER_PATH
        )

        classification_scaler = load_pickle(
            CLASSIFICATION_SCALER_PATH
        )

        classification_model = (
            classification_package["model"]
        )

        classification_threshold = float(
            classification_package.get(
                "decision_threshold",
                0.12
            )
        )

        # Exact training structure:
        # 17 categorical inputs
        # 201 encoded features
        # 14 numerical features
        # 215 final model features

        if classification_encoder.n_features_in_ != 17:
            raise ValueError(
                "Classification encoder expects "
                f"{classification_encoder.n_features_in_} inputs; "
                "expected 17."
            )

        classification_encoded_output = len(
            classification_encoder.get_feature_names_out()
        )

        if classification_encoded_output != 201:
            raise ValueError(
                "Classification encoder outputs "
                f"{classification_encoded_output} features; "
                "expected 201."
            )

        if classification_scaler.n_features_in_ != 14:
            raise ValueError(
                "Classification scaler expects "
                f"{classification_scaler.n_features_in_} features; "
                "expected 14."
            )

        if classification_threshold != 0.12:
            raise ValueError(
                "Classification threshold must be 0.12."
            )

        logger.info(
            "Classification artifacts loaded successfully. Input order: 14 raw numerical + 201 OHE categorical = 215."
        )

    except Exception as exc:

        classification_package = None
        classification_model = None
        classification_encoder = None
        classification_scaler = None

        artifact_errors.append(
            f"Classification artifacts: {exc}"
        )

        logger.exception(
            "Classification artifact loading failed."
        )


# Load artifacts once when API module starts.
load_artifacts()


# ============================================================
# Pydantic: Regression Input
# ============================================================

class RegressionInput(BaseModel):

    age: str
    gender: str
    race: str

    admission_type_id: int
    admission_source_id: int

    medical_specialty: str

    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)

    diag_1: str
    diag_2: str
    diag_3: str

    payer_code: str


# ============================================================
# Pydantic: Classification Input
# ============================================================

class ReadmissionInput(BaseModel):

    age: str
    gender: str
    race: str

    admission_type_id: int
    admission_source_id: int

    time_in_hospital: int = Field(
        ge=1,
        le=14
    )

    medical_specialty: str

    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)

    number_diagnoses: int = Field(ge=0)

    diag_1: str
    diag_2: str
    diag_3: str

    max_glu_serum: str
    A1Cresult: str

    num_medications: int = Field(ge=0)
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)

    insulin: str
    diabetesMed: str
    change: str


# ============================================================
# Regression:
# Exact Missing-Value Handling
# ============================================================

REGRESSION_MISSING_COLUMNS = [
    "race",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3"
]


def apply_regression_missing_handling(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    for column in REGRESSION_MISSING_COLUMNS:

        df[column] = df[column].replace(
            "?",
            "Missing"
        )

    return df


# ============================================================
# Regression:
# Exact Diagnosis Categorization
# ============================================================

def create_regression_diagnosis_category(
    code: str
) -> str:

    code = str(code)

    if code == "Missing":
        return "Missing"

    if code.startswith("250"):
        return "Diabetes"

    if code and code[0].isdigit():

        code_num = float(code)

        if 390 <= code_num <= 459:
            return "Circulatory"

        elif 460 <= code_num <= 519:
            return "Respiratory"

        elif 520 <= code_num <= 579:
            return "Digestive"

        elif 580 <= code_num <= 629:
            return "Genitourinary"

        elif 630 <= code_num <= 679:
            return "Pregnancy"

        elif 680 <= code_num <= 709:
            return "Skin"

        elif 710 <= code_num <= 739:
            return "Musculoskeletal"

        elif 800 <= code_num <= 999:
            return "Injury"

        elif 140 <= code_num <= 239:
            return "Neoplasms"

    if code.startswith("E"):
        return "External_Cause"

    if code.startswith("V"):
        return "Supplementary"

    return "Other"


# ============================================================
# Regression:
# Exact Admission Source Group
# ============================================================

def create_admission_source_group(
    source_id: int
) -> str:

    if source_id in [1, 2, 3]:
        return "Referral"

    elif source_id in [4, 5, 6, 10, 22, 25]:
        return "Transfer"

    elif source_id == 7:
        return "Emergency"

    elif source_id in [
        8, 9, 11, 12, 13, 14, 15,
        17, 18, 19, 20, 21, 23, 24, 26
    ]:
        return "Other"

    else:
        return "Unknown"


# ============================================================
# Regression:
# Exact Final Feature Structure
# ============================================================

REGRESSION_CATEGORICAL_FEATURES = [
    "age",
    "gender",
    "race",
    "admission_type_id",
    "admission_source_id",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "payer_code",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
    "admission_source_group"
]


REGRESSION_NUMERICAL_FEATURES = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "prior_utilization"
]


# ============================================================
# Regression:
# Build Exact Model Input
# ============================================================

def build_regression_model_input(
    data: RegressionInput
):

    df = pd.DataFrame(
        [pydantic_to_dict(data)]
    )

    # Exact notebook missing handling.
    df = apply_regression_missing_handling(df)

    # --------------------------------------------------------
    # prior_utilization
    # --------------------------------------------------------

    df["prior_utilization"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )

    # --------------------------------------------------------
    # inpatient_history_flag
    # --------------------------------------------------------

    df["inpatient_history_flag"] = (
        df["number_inpatient"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Diagnosis categories
    # --------------------------------------------------------

    for column in [
        "diag_1",
        "diag_2",
        "diag_3"
    ]:

        df[f"{column}_category"] = (
            df[column]
            .apply(
                create_regression_diagnosis_category
            )
        )

    # --------------------------------------------------------
    # Admission source group
    # --------------------------------------------------------

    df["admission_source_group"] = (
        df["admission_source_id"]
        .apply(
            create_admission_source_group
        )
    )

    # --------------------------------------------------------
    # Exact categorical order
    # --------------------------------------------------------

    categorical_data = df[
        REGRESSION_CATEGORICAL_FEATURES
    ]

    # --------------------------------------------------------
    # Exact numerical order
    # --------------------------------------------------------

    numerical_data = df[
        REGRESSION_NUMERICAL_FEATURES
    ]

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    encoded = regression_encoder.transform(
        categorical_data
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaled = regression_scaler.transform(
        numerical_data.to_numpy(dtype=np.float32)
    )

    # Convert both representations safely to sparse.
    encoded_sparse = csr_matrix(
        encoded,
        dtype=np.float32
    )

    scaled_sparse = csr_matrix(
        scaled,
        dtype=np.float32
    )

    flag_sparse = csr_matrix(
        df[
            ["inpatient_history_flag"]
        ].to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # Exact training combination:
    #
    # encoded categorical
    # + scaled numerical
    # + inpatient_history_flag
    #
    # = 2359
    # --------------------------------------------------------

    final_matrix = hstack(
        [
            encoded_sparse,
            scaled_sparse,
            flag_sparse
        ],
        format="csr"
    )

    if final_matrix.shape != (1, 2359):

        raise ValueError(
            "Regression model input shape is "
            f"{final_matrix.shape}; expected (1, 2359)."
        )

    return final_matrix


# ============================================================
# Classification:
# Exact Missing-Value Handling
# ============================================================

CLASSIFICATION_MISSING_COLUMNS = [
    "race",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3"
]


def apply_classification_missing_handling(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    for column in CLASSIFICATION_MISSING_COLUMNS:

        df[column] = df[column].replace(
            "?",
            "Missing"
        )

    return df


# ============================================================
# Classification:
# Exact Age Group
# ============================================================

def create_classification_age_group(
    age: str
) -> str:

    if age in [
        "[0-10)",
        "[10-20)"
    ]:
        return "Young"

    elif age in [
        "[20-30)",
        "[30-40)",
        "[40-50)"
    ]:
        return "Adult"

    elif age in [
        "[50-60)",
        "[60-70)"
    ]:
        return "Older Adult"

    elif age in [
        "[70-80)",
        "[80-90)",
        "[90-100)"
    ]:
        return "Elderly"

    else:
        return "Missing"


# ============================================================
# Classification:
# Exact Diagnosis Categorization
# ============================================================

def create_classification_diagnosis_category(
    code: str
) -> str:

    code = str(code).strip()

    if code in [
        "?",
        "Missing",
        "nan",
        "None"
    ]:
        return "Missing"

    if code.startswith(
        ("E", "V")
    ):
        return "External_Cause"

    try:

        numeric_code = float(code)

    except ValueError:

        return "Other"

    if 250 <= numeric_code < 251:
        return "Diabetes"

    elif 1 <= numeric_code < 140:
        return "Infectious"

    elif 140 <= numeric_code < 240:
        return "Neoplasms"

    elif 240 <= numeric_code < 280:
        return "Endocrine_Metabolic"

    elif 280 <= numeric_code < 290:
        return "Blood"

    elif 290 <= numeric_code < 320:
        return "Mental"

    elif 320 <= numeric_code < 360:
        return "Nervous_System"

    elif 390 <= numeric_code < 460:
        return "Circulatory"

    elif 460 <= numeric_code < 520:
        return "Respiratory"

    elif 520 <= numeric_code < 580:
        return "Digestive"

    elif 580 <= numeric_code < 630:
        return "Genitourinary"

    elif 630 <= numeric_code < 680:
        return "Pregnancy"

    elif 680 <= numeric_code < 710:
        return "Skin"

    elif 710 <= numeric_code < 740:
        return "Musculoskeletal"

    elif 740 <= numeric_code < 760:
        return "Congenital"

    elif 760 <= numeric_code < 780:
        return "Perinatal"

    elif 780 <= numeric_code < 800:
        return "Symptoms"

    elif 800 <= numeric_code < 1000:
        return "Injury"

    else:

        return "Other"


# ============================================================
# Classification:
# Exact Final Feature Metadata
# ============================================================

CLASSIFICATION_SELECTED_FEATURES = [
    "age",
    "gender",
    "race",
    "admission_type_id",
    "admission_source_id",
    "time_in_hospital",
    "medical_specialty",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "num_medications",
    "num_lab_procedures",
    "num_procedures",
    "insulin",
    "diabetesMed",
    "change",
    "prior_utilization",
    "high_inpatient_utilization",
    "high_emergency_utilization",
    "diagnosis_complexity",
    "medication_burden_group",
    "age_group",
    "glucose_abnormal_flag",
    "insulin_change_flag",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
    "multiple_prior_encounters_flag"
]


CLASSIFICATION_CATEGORICAL_FEATURES = [
    "age",
    "gender",
    "race",
    "admission_type_id",
    "admission_source_id",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "diabetesMed",
    "change",
    "diagnosis_complexity",
    "medication_burden_group",
    "age_group",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category"
]


CLASSIFICATION_NUMERICAL_FEATURES = [
    "time_in_hospital",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "num_medications",
    "num_lab_procedures",
    "num_procedures",
    "prior_utilization",
    "high_inpatient_utilization",
    "high_emergency_utilization",
    "glucose_abnormal_flag",
    "insulin_change_flag",
    "multiple_prior_encounters_flag"
]


# ============================================================
# Classification:
# Build Exact Model Input
# ============================================================

def build_classification_model_input(
    data: ReadmissionInput
):

    df = pd.DataFrame(
        [pydantic_to_dict(data)]
    )

    # Exact notebook missing handling.
    df = apply_classification_missing_handling(df)

    # --------------------------------------------------------
    # prior_utilization
    # --------------------------------------------------------

    df["prior_utilization"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )

    # --------------------------------------------------------
    # high_inpatient_utilization
    # --------------------------------------------------------

    df["high_inpatient_utilization"] = (
        df["number_inpatient"] >= 2
    ).astype(int)

    # --------------------------------------------------------
    # high_emergency_utilization
    # --------------------------------------------------------

    df["high_emergency_utilization"] = (
        df["number_emergency"] >= 2
    ).astype(int)

    # --------------------------------------------------------
    # diagnosis_complexity
    # --------------------------------------------------------

    df["diagnosis_complexity"] = pd.cut(
        df["number_diagnoses"],
        bins=[
            -1,
            5,
            9,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    # --------------------------------------------------------
    # medication_burden_group
    # --------------------------------------------------------

    df["medication_burden_group"] = pd.cut(
        df["num_medications"],
        # Include zero medications in the Low category.
        bins=[
            -1,
            10,
            20,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    # --------------------------------------------------------
    # age_group
    # --------------------------------------------------------

    df["age_group"] = (
        df["age"]
        .apply(
            create_classification_age_group
        )
    )

    # --------------------------------------------------------
    # glucose_abnormal_flag
    # --------------------------------------------------------

    df["glucose_abnormal_flag"] = (
        df["max_glu_serum"]
        .isin(
            [
                ">200",
                ">300"
            ]
        )
    ).astype(int)

    # --------------------------------------------------------
    # insulin_change_flag
    # --------------------------------------------------------

    df["insulin_change_flag"] = (
        (df["insulin"] != "No")
        &
        (df["change"] == "Ch")
    ).astype(int)

    # --------------------------------------------------------
    # Diagnosis categories
    # --------------------------------------------------------

    for column in [
        "diag_1",
        "diag_2",
        "diag_3"
    ]:

        df[
            f"{column}_category"
        ] = (
            df[column]
            .apply(
                create_classification_diagnosis_category
            )
        )

    # --------------------------------------------------------
    # multiple_prior_encounters_flag
    # --------------------------------------------------------

    df["multiple_prior_encounters_flag"] = (
        df["prior_utilization"] >= 2
    ).astype(int)

    # --------------------------------------------------------
    # Select exact 31 final features
    # --------------------------------------------------------

    work = df[
        CLASSIFICATION_SELECTED_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Exact categorical order
    # --------------------------------------------------------

    categorical_data = work[
        CLASSIFICATION_CATEGORICAL_FEATURES
    ]

    # --------------------------------------------------------
    # Exact numerical order
    # --------------------------------------------------------

    numerical_data = work[
        CLASSIFICATION_NUMERICAL_FEATURES
    ]

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    encoded = classification_encoder.transform(
        categorical_data
    )

    # --------------------------------------------------------
    # Final XGBoost input order
    # --------------------------------------------------------
    # The notebook trains the final XGBoost/Calibrated XGBoost
    # using X_train_encoded = [raw numerical (14), OHE (201)].
    # Do NOT scale the numerical columns here and do NOT reverse
    # the order, otherwise the API can return a wrong prediction
    # while still having shape (1, 215).

    encoded_sparse = csr_matrix(
        encoded,
        dtype=np.float32
    )

    numerical_sparse = csr_matrix(
        numerical_data.to_numpy(dtype=np.float32)
    )

    # --------------------------------------------------------
    # Exact final matrix:
    # 14 raw numerical + 201 one-hot categorical = 215
    # --------------------------------------------------------

    final_matrix = hstack(
        [
            numerical_sparse,
            encoded_sparse
        ],
        format="csr"
    )

    if final_matrix.shape != (1, 215):

        raise ValueError(
            "Classification model input shape is "
            f"{final_matrix.shape}; expected (1, 215)."
        )

    return final_matrix


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def home():

    return {
        "message": (
            "AI Hospital Readmission "
            "& Length-of-Stay API"
        ),
        "status": "running",
        "docs": "/docs"
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():

    regression_loaded = all(
        artifact is not None
        for artifact in [
            regression_model,
            regression_encoder,
            regression_scaler
        ]
    )

    classification_loaded = all(
        artifact is not None
        for artifact in [
            classification_package,
            classification_model,
            classification_encoder,
            classification_scaler
        ]
    )

    healthy = (
        regression_loaded
        and classification_loaded
        and not artifact_errors
    )

    return {
        "status": (
            "healthy"
            if healthy
            else "unhealthy"
        ),
        "regression_loaded": regression_loaded,
        "classification_loaded": classification_loaded,
        "artifact_errors": artifact_errors
    }


# ============================================================
# Regression Prediction Endpoint
# ============================================================

@app.post(
    "/predict/length-of-stay"
)
def predict_length_of_stay(
    data: RegressionInput
):

    if regression_model is None:

        raise HTTPException(
            status_code=500,
            detail="Regression model is not loaded."
        )

    try:

        model_input = (
            build_regression_model_input(
                data
            )
        )

        raw_prediction = float(
            regression_model.predict(
                model_input
            )[0]
        )

        # Exact deployment behavior used in
        # final notebook unseen prediction:
        # restrict to observed target range 1–14.
        prediction = float(
            np.clip(
                raw_prediction,
                1,
                14
            )
        )

        return {
            "prediction": round(
                prediction,
                6
            ),
            "unit": "days",
            "target": "time_in_hospital"
        }

    except Exception as exc:

        logger.exception(
            "Regression prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Regression prediction failed: "
                f"{exc}"
            )
        ) from exc


# ============================================================
# Classification Prediction Endpoint
# ============================================================

@app.post(
    "/predict/readmission"
)
def predict_readmission(
    data: ReadmissionInput
):

    if classification_model is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Classification model "
                "is not loaded."
            )
        )

    try:

        model_input = (
            build_classification_model_input(
                data
            )
        )

        probability = float(
            classification_model.predict_proba(
                model_input
            )[0, 1]
        )

        prediction = int(
            probability >=
            classification_threshold
        )

        return {
            "prediction": prediction,
            "probability": round(
                probability,
                6
            ),
            "threshold": classification_threshold,
            "target": "target_readmit_30d",
            "readmission": (
                "30-day readmission risk"
                if prediction == 1
                else "No 30-day readmission risk"
            )
        }

    except Exception as exc:

        logger.exception(
            "Classification prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Classification prediction failed: "
                f"{exc}"
            )
        ) from exc
