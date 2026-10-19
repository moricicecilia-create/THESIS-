# MESSIDOR: REFERENCE CONDITION
# Models: TabPFN-v2, TabICLv2, TabDPT, CatBoost, MLP (no Mitra).
# No artificial missingness or post-hoc calibration.
# Splits: 60% training/context, 10% validation, 15% calibration, 15% test.
# Target: 0 = no signs of diabetic retinopathy, 1 = signs of diabetic retinopathy.
# 1151 observations and 19 numeric features. One stratified split (seed 42).

from pathlib import Path
import time
import shutil
import gc

import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    log_loss,
    brier_score_loss,
)

from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabicl import TabICLClassifier
from autogluon.tabular import TabularPredictor

RANDOM_STATE = 42
CPU_THREADS = 4
DATASET_FILE = Path(__file__).resolve().parent / "messidor_features.arff"
RESULTS_DIR = Path(__file__).resolve().parent / "results_messidor"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("1. LOAD DATASET")
print("=" * 70)
if not DATASET_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_FILE.resolve()}")

# ARFF contains 19 numeric attributes and the binary Class label.
raw, metadata = arff.loadarff(DATASET_FILE)
df_raw = pd.DataFrame(raw)
feature_cols = [str(i) for i in range(19)]

print("Original shape:", df_raw.shape)
print("\n" + "=" * 70)
print("2. DATA QUALITY CHECK")
print("=" * 70)
print("Observations:", len(df_raw))
print("Features:", len(feature_cols))
print("Missing values:", int(df_raw.isna().sum().sum()))
print("Duplicate rows:", int(df_raw.duplicated().sum()))
print("\nOriginal class distribution:")
print(df_raw["Class"].value_counts())

if df_raw.isna().any().any():
    raise ValueError("Unexpected missing values in Messidor dataset.")
if set(df_raw["Class"].unique()) != {b"0", b"1"}:
    raise ValueError("Unexpected class labels; expected 0 and 1.")

# Keep all observations; original duplicate rows are reported, not removed.
# Convert ARFF byte labels into integers, and keep the 19 measurements as inputs.
df = df_raw[feature_cols].copy()
df.columns = [f"feature_{i + 1}" for i in range(19)]
df["target"] = df_raw["Class"].map({b"0": 0, b"1": 1}).astype(int)

print("\n" + "=" * 70)
print("3. BINARY TARGET")
print("=" * 70)
print("0 = No signs of diabetic retinopathy; 1 = Signs of diabetic retinopathy")
print(df["target"].value_counts().sort_index())
print("Model input shape:", df.shape)

X = df.drop(columns="target")
y = df["target"]


print("\n" + "=" * 70)

print("4. STRATIFIED SPLIT")

print("=" * 70)


# Stratify all four partitions to preserve the class proportions.
all_indices = np.arange(len(df)) 


train_cal_val_idx, test_idx = train_test_split(

    all_indices,

    test_size=0.15,

    stratify=y,

    random_state=RANDOM_STATE

)


train_val_idx, calibration_idx = train_test_split(

    train_cal_val_idx,

    test_size=0.15 / 0.85,

    stratify=y.iloc[train_cal_val_idx], 

    random_state=RANDOM_STATE

)


train_idx, validation_idx = train_test_split(

    train_val_idx,

    test_size=0.10 / 0.70,

    stratify=y.iloc[train_val_idx],

    random_state=RANDOM_STATE

)


X_train = X.iloc[train_idx].copy()

y_train = y.iloc[train_idx].copy()


X_val = X.iloc[validation_idx].copy()

y_val = y.iloc[validation_idx].copy()


X_cal = X.iloc[calibration_idx].copy()

y_cal = y.iloc[calibration_idx].copy()


X_test = X.iloc[test_idx].copy()

y_test = y.iloc[test_idx].copy()


print("\nDimensioni:")


print(

    "TRAIN / CONTEXT:",

    X_train.shape

)


print(

    "VALIDATION:",

    X_val.shape

)


print(

    "CALIBRATION:",

    X_cal.shape

)


print(

    "TEST:",

    X_test.shape

)


def print_class_distribution(name, y_subset):

    counts = y_subset.value_counts().sort_index() 

    proportions = (

        y_subset.value_counts(

            normalize=True

        ).sort_index()

    )


    print(f"\n{name}") 


    print(

        "Class 0 (no signs):",

        counts.get(0, 0), 
        f"({proportions.get(0, 0) * 100:.2f}%)" 

    )


    print(

        "Class 1 (signs):",

        counts.get(1, 0),

        f"({proportions.get(1, 0) * 100:.2f}%)"

    )


print_class_distribution(

    "FULL DATASET",

    y

)


print_class_distribution(

    "TRAIN / CONTEXT",

    y_train

)


print_class_distribution(

    "VALIDATION",

    y_val

)


print_class_distribution(

    "CALIBRATION",

    y_cal

)


print_class_distribution(

    "TEST",

    y_test

)


split_info = pd.DataFrame({ 

    "original_index": df.index,

    "split": "unknown"

})


split_info.loc[train_idx, "split"] = "train_context" 

split_info.loc[validation_idx, "split"] = "validation"

split_info.loc[calibration_idx, "split"] = "calibration"

split_info.loc[test_idx, "split"] = "test"


split_info.to_csv(

    RESULTS_DIR / "messidor_split.csv",

    index=False

)


# Confidence-based ECE with ten equally spaced bins.
def expected_calibration_error(

    y_true,

    probabilities,

    n_bins=10 ,

):

    """Compute the weighted gap between bin accuracy and mean confidence."""


    y_true = np.asarray(y_true) 
    probabilities = np.asarray(probabilities)


    predictions = (

        probabilities >= 0.5 

    ).astype(int) 
    

    confidence = np.where(

        predictions == 1,

        probabilities,

        1 - probabilities

    )


    bin_edges = np.linspace( 

        0.0,

        1.0,

        n_bins + 1

    )
    


    ece = 0.0


    for i in range(n_bins): 
    

        lower = bin_edges[i]

        upper = bin_edges[i + 1]


        if i == n_bins - 1:

            mask = (

                (confidence >= lower)

                &

                (confidence <= upper)

            )

        else:

            mask = (

                (confidence >= lower)

                &

                (confidence < upper)

            )


        if np.sum(mask) == 0:

            continue


        
        bin_accuracy = np.mean(

            predictions[mask]

            == y_true[mask]

        )


        bin_confidence = np.mean(

            confidence[mask]

        )


        bin_fraction = (

            np.sum(mask)

            / len(y_true)

        )


        ece += (

            bin_fraction 

            * abs( 

                bin_accuracy

                - bin_confidence

            ) 

        )


    return ece


# Evaluate discrimination, classification accuracy and probability quality.
def calculate_metrics(

    y_true, 

    probabilities 

):


    predictions = (

        probabilities >= 0.5

    ).astype(int) 


    return { 

        "AUROC": roc_auc_score(

            y_true,

            probabilities

        ),
        


        "Balanced_Accuracy":

            balanced_accuracy_score(

                y_true,

                predictions

            ),


        "Log_Loss":

            log_loss(

                y_true,

                probabilities,

                labels=[0, 1]

            ),


        "Brier_Score":

            brier_score_loss(

                y_true,

                probabilities

            ),


        "ECE":

            expected_calibration_error(

                y_true,

                probabilities

            )

    }


results = []

model_statuses = []


probability_results = pd.DataFrame({

    "y_calibration": pd.Series(y_cal.to_numpy()), 

    "y_test": pd.Series(y_test.to_numpy())

}) 


# Store held-out test metrics and uncalibrated probabilities.
def save_model_result(

    model_name,

    probabilities_cal,

    probabilities_test,

    fit_time,

    prediction_time

):


    metrics = calculate_metrics(

        y_test,

        probabilities_test

    )


    row = {

        "model": model_name,

        "fit_time_seconds": fit_time,

        "prediction_time_seconds":

            prediction_time,

        **metrics

    }


    results.append(row) 

    model_statuses.append({
        "model": model_name,
        "status": "success",
        "error": ""
    })


    probability_results[

        f"{model_name}_calibration"

    ] = probabilities_cal


    probability_results[

        f"{model_name}_test"

    ] = probabilities_test


    print("\n" + "-" * 60)

    print(model_name)

    print("-" * 60)


    print(

        f"Fit time: {fit_time:.2f} s"

    )


    print(

        f"Prediction time: "

        f"{prediction_time:.2f} s"

    )


    for metric, value in metrics.items():


        print(

            f"{metric}: {value:.4f}"

        )


print("\n" + "=" * 70)

print("5. TABPFN-v2")

print("=" * 70)


try:


    start_fit = time.perf_counter()


    


    tabpfn = (

        TabPFNClassifier

        .create_default_for_version(

            ModelVersion.V2,

            device="cpu", 

            n_estimators=8, 

            random_state=RANDOM_STATE, 

            n_preprocessing_jobs=1,

            show_progress_bar=False

        )

    )


    tabpfn.fit(

        X_train,

        y_train

    )


    fit_time = (

        time.perf_counter()

        - start_fit

    )


    start_prediction = (

        time.perf_counter()

    )


    proba_cal = (

        tabpfn

        .predict_proba(X_cal)[:, 1]

    )


    proba_test = (

        tabpfn

        .predict_proba(X_test)[:, 1]

    )


    prediction_time = (

        time.perf_counter()

        - start_prediction

    )


    save_model_result(

        "TabPFN-v2",

        proba_cal,

        proba_test,

        fit_time,

        prediction_time

    )


    del tabpfn 

    gc.collect() 


except Exception as e:


    print("\nERRORE TABPFN:")

    print(type(e).__name__)

    print(e)

    model_statuses.append({
        "model": "TabPFN-v2",
        "status": "failed",
        "error": f"{type(e).__name__}: {e}"
    })

    gc.collect()


print("\n" + "=" * 70)

print("6. TABICLv2")

print("=" * 70)


try:


    start_fit = time.perf_counter()


    tabicl = TabICLClassifier(

        checkpoint_version=

            "tabicl-classifier-v2-20260212.ckpt", 


        device="cpu",


        n_estimators=8,


        n_jobs=CPU_THREADS, 


        random_state=RANDOM_STATE,


        verbose=False

    )


    tabicl.fit(

        X_train,

        y_train

    )


    fit_time = (

        time.perf_counter()

        - start_fit

    ) 


    start_prediction = (

        time.perf_counter()

    ) 


    proba_cal = (

        tabicl

        .predict_proba(X_cal)[:, 1] 

    )


    proba_test = (

        tabicl

        .predict_proba(X_test)[:, 1]

    )


    prediction_time = (

        time.perf_counter()

        - start_prediction

    )


    save_model_result(

        "TabICLv2",

        proba_cal,

        proba_test,

        fit_time,

        prediction_time

    )


    del tabicl

    gc.collect()


except Exception as e:


    print("\nERRORE TABICL:")

    print(type(e).__name__)

    print(e)

    model_statuses.append({
        "model": "TabICLv2",
        "status": "failed",
        "error": f"{type(e).__name__}: {e}"
    })

    gc.collect()


print("\n" + "=" * 70)

print("7. TABDPT")

print("=" * 70)


try:


    tabdpt_path = (

        RESULTS_DIR

        / "tabdpt_model"

    )


    if tabdpt_path.exists():

        shutil.rmtree(tabdpt_path)


    train_tabdpt = X_train.copy() 

    train_tabdpt["target"] = y_train.to_numpy() 


    val_tabdpt = X_val.copy()

    val_tabdpt["target"] = y_val.to_numpy()


    cal_tabdpt = X_cal.copy()

    cal_tabdpt["target"] = y_cal.to_numpy()


    test_tabdpt = X_test.copy()

    test_tabdpt["target"] = y_test.to_numpy()


    start_fit = time.perf_counter()


    tabdpt_predictor = TabularPredictor(

        label="target",

        problem_type="binary",

        path=str(tabdpt_path),

        verbosity=2 

    )

    

    tabdpt_predictor.fit(

        train_data=train_tabdpt,

        tuning_data=val_tabdpt,

        hyperparameters={

            "TABDPT": [{}] 

        },

        num_gpus=0,

        num_cpus=CPU_THREADS,

        fit_weighted_ensemble=False, 

        dynamic_stacking=False,

        raise_on_model_failure=True

    )


    fit_time = (

        time.perf_counter()

        - start_fit

    )


    start_prediction = time.perf_counter()


    proba_cal_df = tabdpt_predictor.predict_proba(

        cal_tabdpt

    )

    proba_test_df = tabdpt_predictor.predict_proba(

        test_tabdpt

    )


    proba_cal = np.asarray(proba_cal_df)[:, 1] 

    proba_test = np.asarray(proba_test_df)[:, 1]


    prediction_time = (

        time.perf_counter()

        - start_prediction

    )


    save_model_result(

        "TabDPT",

        proba_cal,

        proba_test,

        fit_time,

        prediction_time

    )


    del tabdpt_predictor

    gc.collect()


except Exception as e:


    print("\nERRORE TABDPT:")

    print(type(e).__name__)

    print(e)

    model_statuses.append({
        "model": "TabDPT",
        "status": "failed",
        "error": f"{type(e).__name__}: {e}"
    })

    gc.collect()


print("\n" + "=" * 70)
print("8. CATBOOST")

print("=" * 70)


try:


    catboost_model = CatBoostClassifier(


        iterations=500, 


        learning_rate=0.03,


        depth=6, 


        l2_leaf_reg=3, 


        loss_function="Logloss", 


        eval_metric="AUC",


        random_seed=RANDOM_STATE,


        thread_count=CPU_THREADS,


        verbose=False

    )


    start_fit = time.perf_counter()


    catboost_model.fit(


        X_train,

        y_train,


        eval_set=(

            X_val,

            y_val

        ),


        use_best_model=True, 


        early_stopping_rounds=50, 


        verbose=False

    )


    fit_time = (

        time.perf_counter()

        - start_fit

    )


    start_prediction = (

        time.perf_counter()

    )


    proba_cal = (

        catboost_model

        .predict_proba(X_cal)[:, 1]

    )


    proba_test = (

        catboost_model

        .predict_proba(X_test)[:, 1]

    )


    prediction_time = (

        time.perf_counter()

        - start_prediction

    )


    save_model_result(

        "CatBoost",

        proba_cal,

        proba_test,

        fit_time,

        prediction_time

    )


    del catboost_model

    gc.collect()


except Exception as e:


    print("\nERRORE CATBOOST:")

    print(type(e).__name__)

    print(e)

    model_statuses.append({
        "model": "CatBoost",
        "status": "failed",
        "error": f"{type(e).__name__}: {e}"
    })

    gc.collect()


print("\n" + "=" * 70)

print("9. MLP")

print("=" * 70)


try:


    # StandardScaler is fitted on the training data only.
    mlp_model = Pipeline([


        (
            "scaler",

            StandardScaler() 

        ),


        
        (

            "mlp",

            MLPClassifier(


                hidden_layer_sizes=(

                    64,

                    32

                ),


                activation="relu",


                solver="lbfgs",


                alpha=1e-4,


                max_iter=1000,


                random_state=RANDOM_STATE

            )

        )

    ])


    start_fit = time.perf_counter()


    mlp_model.fit(

        X_train,

        y_train

    )


    fit_time = (

        time.perf_counter()

        - start_fit

    )


    start_prediction = (

        time.perf_counter()

    )


    proba_cal = (

        mlp_model

        .predict_proba(X_cal)[:, 1]

    )


    proba_test = (

        mlp_model

        .predict_proba(X_test)[:, 1]

    )


    prediction_time = (

        time.perf_counter()

        - start_prediction

    )


    save_model_result(

        "MLP",

        proba_cal,

        proba_test,

        fit_time,

        prediction_time

    )


    del mlp_model

    gc.collect()


except Exception as e:


    print("\nERRORE MLP:")

    print(type(e).__name__)

    print(e)

    model_statuses.append({
        "model": "MLP",
        "status": "failed",
        "error": f"{type(e).__name__}: {e}"
    })

    gc.collect()


print("\n" + "=" * 70)

print("21. SAVE RESULTS")

print("=" * 70)


results_df = pd.DataFrame(

    results

)


results_df.to_csv(

    RESULTS_DIR

    / "messidor_reference_results.csv",

    index=False

)


probability_results.to_csv(

    RESULTS_DIR

    / "messidor_reference_probabilities.csv",

    index=False

)


status_df = pd.DataFrame(model_statuses)


status_df.to_csv(

    RESULTS_DIR

    / "messidor_model_status.csv",

    index=False

)


print("\n" + "=" * 70)

print("FINAL RESULTS")

print("=" * 70)


if len(results_df) > 0:


    print(

        results_df.to_string(

            index=False

        )

    )


else:


    print(

        "No model produced results."

    )


print("\n" + "=" * 70)

print("PIPELINE COMPLETED")

print("=" * 70)


print(

    "\nResults saved in:"

)


print(

    RESULTS_DIR.resolve()

)


print(

    "\nFiles created:"

)


print(

    "- messidor_split.csv"

)


print(

    "- messidor_reference_results.csv"

)


print(

    "- messidor_reference_probabilities.csv"

)


print(

    "- messidor_model_status.csv"

)


print(
    "\nReference condition completed."
)


