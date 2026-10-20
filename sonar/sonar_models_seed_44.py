# SONAR binary classification benchmark
# Five evaluated models: TabPFN-v2, TabICLv2, TabDPT, CatBoost and MLP.
# The same seed controls stratified splits and synthetic missingness masks.
# Synthetic missingness is introduced only into calibration and test data.
# Predictions and metrics are saved to a seed-specific results directory.
# Temperature scaling is performed separately after probability export.

# ============================================================


# TabPFN-v2 | TabICLv2 | Mitra-v2 (prova) | Mitra-v1 (prova) | TabDPT | CatBoost | MLP

#


# ============================================================


# ============================================================

# 1. IMPORTS

# ============================================================


from pathlib import Path


import time


import shutil


import gc
import os  


import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer


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


# ============================================================

# 2. CONFIGURATION

# ============================================================


RANDOM_STATE = 44 


# Numero di thread CPU usato dove i framework lo supportano.

CPU_THREADS = 4


DATASET_DIR = Path(__file__).resolve().parent 


DATASET_FILE = DATASET_DIR / "sonar.all-data"


RESULTS_DIR = DATASET_DIR / "results_sonar" / "seed_44"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================

# 3. LOAD DATASET

# ============================================================


print("\n" + "=" * 70)

print("1. LOAD DATASET")

print("=" * 70)


if not DATASET_FILE.exists():

    raise FileNotFoundError(

        f"Dataset non trovato:\n{DATASET_FILE.resolve()}"

    )


df = pd.read_csv(

    DATASET_FILE,

    header=None
)


df.columns = (

    [f"feature_{i}" for i in range(1, 61)]

    + ["target"]

)


print("Dataset loaded successfully.")

print("Shape:", df.shape)


# ============================================================

# 4. DATASET VALIDATION

# ============================================================


print("\n" + "=" * 70)

print("2. DATASET VALIDATION")

print("=" * 70)


print("\nObservations:", len(df))

print("Feature:", len(df.columns) - 1)

print("Total columns:", len(df.columns))


print(

    "\nMissing values:",

    df.isnull().sum().sum()

)


print(

    "Duplicati:",

    df.duplicated().sum()

)


print(
    "\nDistribuzione target:"
)


print(

    df["target"].value_counts()

)


valid_classes = {"M", "R"}


if set(df["target"].unique()) != valid_classes:

    raise ValueError(

        f"Classi inattese: {df['target'].unique()}"

    )


# ============================================================

# 5. TARGET ENCODING

# ============================================================


print("\n" + "=" * 70)

print("3. TARGET ENCODING")

print("=" * 70)


df["target"] = df["target"].map({

    "R": 0,

    "M": 1

})


print(

    "\nDistribuzione dopo conversione:"

)


print(

    df["target"].value_counts()

)


# ============================================================

# 6. SEPARATE FEATURES AND LABELS

# ============================================================


X = df.drop(columns="target")


y = df["target"]


# ============================================================

# 7. STRATIFIED DATA SPLIT

# ============================================================


print("\n" + "=" * 70)

print("4. STRATIFIED SPLIT")

print("=" * 70)


#


#


all_indices = np.arange(len(df)) 


# ------------------------------------------------------------


# ------------------------------------------------------------


train_cal_val_idx, test_idx = train_test_split(

    all_indices,

    test_size=0.15,

    stratify=y,

    random_state=RANDOM_STATE

)


# ------------------------------------------------------------


#


#


# ------------------------------------------------------------


train_val_idx, calibration_idx = train_test_split(

    train_cal_val_idx,

    test_size=0.15 / 0.85,

    stratify=y.iloc[train_cal_val_idx], 

    random_state=RANDOM_STATE

)


# ------------------------------------------------------------


#

# ------------------------------------------------------------


train_idx, validation_idx = train_test_split(

    train_val_idx,

    test_size=0.10 / 0.70,

    stratify=y.iloc[train_val_idx],

    random_state=RANDOM_STATE

)


# ============================================================

# 8. CREATE FOUR DATA SUBSETS

# ============================================================


X_train = X.iloc[train_idx].copy()

y_train = y.iloc[train_idx].copy()


X_val = X.iloc[validation_idx].copy()

y_val = y.iloc[validation_idx].copy()


X_cal = X.iloc[calibration_idx].copy()

y_cal = y.iloc[calibration_idx].copy()


X_test = X.iloc[test_idx].copy()

y_test = y.iloc[test_idx].copy()


print("\nSubset sizes:")


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


# ============================================================

# 9. CHECK CLASS STRATIFICATION

# ============================================================


def print_class_distribution(name, y_subset):


    
    
    
    counts = y_subset.value_counts().sort_index() 
    

    
    
    proportions = (

        y_subset.value_counts(

            normalize=True

        ).sort_index()

    )


    print(f"\n{name}") 


    print(

        "Classe 0 (Rock):",

        counts.get(0, 0), 
        
        f"({proportions.get(0, 0) * 100:.2f}%)" 

    )


    
    print(

        "Classe 1 (Mine):",

        counts.get(1, 0),

        f"({proportions.get(1, 0) * 100:.2f}%)"

    )


print_class_distribution(

    "DATASET COMPLETO",

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


# ============================================================
# 9 BIS. GENERATE MISSINGNESS SCENARIOS (PREPARATION ONLY)
# ============================================================


MISSING_RATES = (0.10, 0.30, 0.50)


def introduce_missingness(X_input, rate, pattern, seed):
    """SONAR experimental helper. See the English README for the pipeline details."""

    
    X_masked = X_input.copy(deep=True)

    
    rng = np.random.default_rng(seed)

    n_rows, n_cols = X_masked.shape

    if pattern == "cell":

        
        
        mask = rng.random((n_rows, n_cols)) < rate

        X_masked = X_masked.mask(mask)

    elif pattern == "feature":

        
        
        n_missing_cols = round(rate * n_cols)

        cols_to_mask = rng.choice(
            n_cols,
            size=n_missing_cols,
            replace=False
        )

        X_masked.iloc[:, cols_to_mask] = np.nan

    else:
        raise ValueError("pattern must be 'cell' or 'feature'")

    return X_masked


X_cal_scenarios = {"reference": X_cal.copy(deep=True)}
X_test_scenarios = {"reference": X_test.copy(deep=True)}


for pattern in ("cell", "feature"):

    for rate in MISSING_RATES:

        scenario_name = f"{pattern}_{int(rate * 100)}"

        
        
        pattern_offset = 0 if pattern == "cell" else 1000
        rate_offset = int(rate * 100)

        X_cal_scenarios[scenario_name] = introduce_missingness(
            X_cal,
            rate,
            pattern,
            seed=RANDOM_STATE + 100 + pattern_offset + rate_offset
        )

        X_test_scenarios[scenario_name] = introduce_missingness(
            X_test,
            rate,
            pattern,
            seed=RANDOM_STATE + 200 + pattern_offset + rate_offset
        )


assert len(X_cal_scenarios) == 7
assert len(X_test_scenarios) == 7

assert X_cal_scenarios["reference"].equals(X_cal)
assert X_test_scenarios["reference"].equals(X_test)

assert not X_train.isna().any().any()


print("\nMISSINGNESS SCENARIOS GENERATED (used in the following model sections):")

for scenario_name, X_scenario in X_test_scenarios.items():

    observed_rate = X_scenario.isna().to_numpy().mean() * 100

    print(
        f"{scenario_name:>12}: "
        f"{observed_rate:5.1f}% di celle mancanti sul TEST"
    )


# ============================================================
# 9 TER. IMPUTATION FOR MODELS THAT REQUIRE IT
# ============================================================


imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)


imputer.fit(X_train)


# ============================================================

# 10. SAVE DATA SPLIT

# ============================================================


split_info = pd.DataFrame({ 

    "original_index": df.index,

    "split": "unknown"

})


split_info.loc[train_idx, "split"] = "train_context" 


split_info.loc[validation_idx, "split"] = "validation"

split_info.loc[calibration_idx, "split"] = "calibration"

split_info.loc[test_idx, "split"] = "test"


split_info.to_csv(

    RESULTS_DIR / "sonar_split.csv",

    index=False

)


# ============================================================

# 11. EVALUATION METRICS

# ============================================================


def expected_calibration_error(

    y_true,

    probabilities,

    n_bins=10 ,

):

    """SONAR experimental helper. See the English README for the pipeline details."""


    
    
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

            bin_fraction #indica quanto è grande quel bin rispetto a tutto il dataset 

            * abs( 

                bin_accuracy

                - bin_confidence

            ) 

        )


    return ece


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


# ============================================================

# 12. RESULT CONTAINERS

# ============================================================


results = []


model_statuses = []


# ============================================================

# ============================================================


missingness_probabilities = []


def save_scenario_probabilities(
    model_name,
    scenario_name,
    subset_name,
    y_true,
    probabilities
):

    
    y_array = np.asarray(y_true)
    p_array = np.asarray(probabilities)

    
    
    if len(y_array) != len(p_array):
        raise ValueError(
            "Numero di probabilità diverso dal numero di target."
        )

    
    
    if not np.isfinite(p_array).all() or np.any((p_array < 0) | (p_array > 1)):
        raise ValueError("Invalid probabilities: NaN, infinity, or values outside [0, 1].")

    
    scenario_df = pd.DataFrame({
        "model": model_name,
        "scenario": scenario_name,
        "subset": subset_name,
        "observation_id": y_true.index.to_numpy(),
        "y_true": y_array,
        "probability": p_array
    })

    
    missingness_probabilities.append(scenario_df)


def save_checkpoint_missingness(model_name=None, model_rows=None):
    """SONAR experimental helper. See the English README for the pipeline details."""

    def write_csv_atomic(dataframe, destination):
        temporary = destination.with_name(destination.name + ".tmp")
        dataframe.to_csv(temporary, index=False)
        os.replace(temporary, destination)

    if missingness_probabilities:
        all_probabilities = pd.concat(
            missingness_probabilities, ignore_index=True
        )
        write_csv_atomic(
            all_probabilities,
            RESULTS_DIR / "sonar_missingness_probabilities.csv"
        )

    
    write_csv_atomic(
        pd.DataFrame(model_statuses),
        RESULTS_DIR / "sonar_model_status.csv"
    )

    
    if model_name is not None and model_rows is not None:
        write_csv_atomic(
            pd.DataFrame(model_rows),
            RESULTS_DIR / f"{model_name}_missingness_results.csv"
        )


probability_results = pd.DataFrame({

    "y_calibration": y_cal.to_numpy(), 

    "y_test": y_test.to_numpy()

}) 


# ============================================================

# 13. GENERAL FUNCTION TO SAVE RESULTS

# ============================================================


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


# ============================================================
# 14. TABPFN-v2
# ============================================================


print("\n" + "=" * 70)

print("5. TABPFN-v2")

print("=" * 70)


try:


    start_fit = time.perf_counter()


    

    # il progetto richiede TabPFN-v2.

    


    
    
    
    # CPU e seed fissano risorse e casualità dell'esperimento.
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


    # ============================================================
    
    # ============================================================

    
    
    
    

    
    tabpfn_missing_results = []

    
    
    # - reference: dataset completo;
    
    

    for scenario_name, X_scenario in X_test_scenarios.items():

        
        
        probability_checkpoint_length = len(missingness_probabilities)
        try:
            print(f"\nTABPFN-v2 - Scenario: {scenario_name}")

            
            
            
            

            start_missing_prediction = time.perf_counter()

       
            
            
            

            if scenario_name == "reference":
                probabilities = proba_test

            else:
                
                
                probabilities = tabpfn.predict_proba(
                    X_scenario
                )[:, 1]

 
            # ========================================================
            
            # ========================================================

            
            
            
            

            save_scenario_probabilities(
                model_name="TabPFN-v2",
                scenario_name=scenario_name,
                subset_name="test",
                y_true=y_test,
                probabilities=probabilities
            )

            
            
            

            X_cal_scenario = X_cal_scenarios[scenario_name]

        
            
            

            if scenario_name == "reference":
                probabilities_cal = proba_cal

            else:
                
                
                probabilities_cal = tabpfn.predict_proba(
                    X_cal_scenario
                )[:, 1]


            
            
            

            save_scenario_probabilities(
                model_name="TabPFN-v2",
                scenario_name=scenario_name,
                subset_name="calibration",
                y_true=y_cal,
                probabilities=probabilities_cal
            )

       
            
            
            
            

            if scenario_name == "reference":
                missing_prediction_time = prediction_time

            else:
                missing_prediction_time = (
                    time.perf_counter() - start_missing_prediction
                )


            
            
            metrics = calculate_metrics(
                y_test,
                probabilities
            )

            
            
            row = {
                "model": "TabPFN-v2",
                "scenario": scenario_name,
                "prediction_time_seconds": missing_prediction_time,
                **metrics
            }

            
            tabpfn_missing_results.append(row)

            
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            print(
                f"Prediction time: {missing_prediction_time:.2f} s"
            )

            
            model_statuses.append({
                "model": "TabPFN-v2",
                "scenario": scenario_name,
                "status": "scenario_success",
                "error": ""
            })
            save_checkpoint_missingness("tabpfn", tabpfn_missing_results)

        except Exception as scenario_error:
            
            
            del missingness_probabilities[probability_checkpoint_length:]
            print(f"\nERROR TabPFN-v2, scenario {scenario_name}:")
            print(type(scenario_error).__name__, scenario_error)
            model_statuses.append({
                "model": "TabPFN-v2",
                "scenario": scenario_name,
                "status": "scenario_failed",
                "error": f"{type(scenario_error).__name__}: {scenario_error}"
            })
            save_checkpoint_missingness("tabpfn", tabpfn_missing_results)
            continue

    
    tabpfn_missing_df = pd.DataFrame(
        tabpfn_missing_results
    )

    
    tabpfn_missing_df.to_csv(
        RESULTS_DIR / "tabpfn_missingness_results.csv",
        index=False
    )

    print("\nTABPFN-v2 - MISSINGNESS EVALUATION COMPLETED")


 # Abbiamo terminato tutte le predizioni di TabPFN,
    
    
    del tabpfn
    gc.collect()


except Exception as e:

    # Se si verifica un errore durante l'esecuzione di TabPFN,
    
    print("\nERROR TABPFN:")
    print(type(e).__name__)
    print(e)

    model_statuses.append({
        "model": "TabPFN-v2",
        "status": "failed",
        "error": f"{type(e).__name__}: {e}"
    })

    gc.collect()


# ============================================================

# 15. TABICLv2

# ============================================================


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


    # ============================================================
    
    # ============================================================

    
    
    
    

    
    tabicl_missing_results = []

    
    
    

    for scenario_name, X_scenario in X_test_scenarios.items():

        
        
        probability_checkpoint_length = len(missingness_probabilities)
        try:
            print(f"\nTABICLv2 - Scenario: {scenario_name}")

            
            
            

            
            start_missing_prediction = time.perf_counter()

       
            
            
            

            if scenario_name == "reference":
                probabilities = proba_test

            else:
                
                
                probabilities = tabicl.predict_proba(
                    X_scenario
                )[:, 1]

            # ========================================================
            
            # ========================================================

            
            
            

            save_scenario_probabilities(
                model_name="TabICLv2",
                scenario_name=scenario_name,
                subset_name="test",
                y_true=y_test,
                probabilities=probabilities
            )

            
            
            

            X_cal_scenario = X_cal_scenarios[scenario_name]

        
            
            

            if scenario_name == "reference":
                probabilities_cal = proba_cal

            else:
                
                
                probabilities_cal = tabicl.predict_proba(
                    X_cal_scenario
                )[:, 1]


            
            
            

            save_scenario_probabilities(
                model_name="TabICLv2",
                scenario_name=scenario_name,
                subset_name="calibration",
                y_true=y_cal,
                probabilities=probabilities_cal
            )

        
            
            

            if scenario_name == "reference":
                missing_prediction_time = prediction_time

            else:
                
                
                missing_prediction_time = (
                    time.perf_counter() - start_missing_prediction
                )


            
            metrics = calculate_metrics(
                y_test,
                probabilities
            )

            
            row = {
                "model": "TabICLv2",
                "scenario": scenario_name,
                "prediction_time_seconds": missing_prediction_time,
                **metrics
            }

            
            tabicl_missing_results.append(row)

            
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            print(
                f"Prediction time: {missing_prediction_time:.2f} s"
            )

            
            model_statuses.append({
                "model": "TabICLv2",
                "scenario": scenario_name,
                "status": "scenario_success",
                "error": ""
            })
            save_checkpoint_missingness("tabicl", tabicl_missing_results)

        except Exception as scenario_error:
            
            
            del missingness_probabilities[probability_checkpoint_length:]
            print(f"\nERROR TabICLv2, scenario {scenario_name}:")
            print(type(scenario_error).__name__, scenario_error)
            model_statuses.append({
                "model": "TabICLv2",
                "scenario": scenario_name,
                "status": "scenario_failed",
                "error": f"{type(scenario_error).__name__}: {scenario_error}"
            })
            save_checkpoint_missingness("tabicl", tabicl_missing_results)
            continue

    
    tabicl_missing_df = pd.DataFrame(
        tabicl_missing_results
    )

    
    tabicl_missing_df.to_csv(
        RESULTS_DIR / "tabicl_missingness_results.csv",
        index=False
    )

    print("\nTABICLv2 - MISSINGNESS EVALUATION COMPLETED")


    del tabicl

    gc.collect()


except Exception as e:


    print("\nERROR TABICL:")

    print(type(e).__name__)

    print(e)

    model_statuses.append({
        "model": "TabICLv2",
        "status": "failed",
        "error": f"{type(e).__name__}: {e}"
    })

    gc.collect()


# ============================================================

# 16. MITRA-v2

# ============================================================


# print("7. MITRA-v2")


#             "MITRA": {


#                     "autogluon/mitra-classifier-2",


#         # Impone l'esecuzione CPU e limita le risorse; gli ensemble sono


#         "Mitra-v2",


#     print("\nERROR MITRA-v2:")


#         "model": "Mitra-v2",


# # ============================================================

# # 17. MITRA-v1

# # ============================================================


# # a MITRA-v2. Mantenere uguali split, risorse e protocollo aiuta ad attribuire


# print("8. MITRA-v1")


#             "MITRA": {


#                     "autogluon/mitra-classifier",


#         "Mitra-v1",


#     print("\nERROR MITRA-v1:")


#         "model": "Mitra-v1",


# ============================================================

# 18. TABDPT

# ============================================================


print("\n" + "=" * 70)

print("9. TABDPT")

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

        # Chiede ad AutoGluon di usare TabDPT senza aggiungere modelli diversi.
        
        # combinazione di modelli anziché la prestazione di TabDPT.
        hyperparameters={

            "TABDPT": [{}] #usa tabdpt senza modificare parametri

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


    # ============================================================
    
    # ============================================================

    # TabDPT è già stato addestrato sul training set completo.
    
    

    
    tabdpt_missing_results = []

    
    # Sono gli stessi utilizzati da TabPFN e TabICL.
    for scenario_name, X_scenario in X_test_scenarios.items():

        
        
        probability_checkpoint_length = len(missingness_probabilities)
        try:
            print(f"\nTABDPT - Scenario: {scenario_name}")

            
            
            
            

            X_scenario_tabdpt = X_scenario.copy()

            
            start_missing_prediction = time.perf_counter()

        
            
            

            if scenario_name == "reference":
                probabilities = proba_test

            else:
                
                
                

                proba_scenario_df = tabdpt_predictor.predict_proba(
                    X_scenario_tabdpt
                )

                
                probabilities = np.asarray(
                    proba_scenario_df
                )[:, 1]


            # ========================================================
            
            # ========================================================

            
            
            

            save_scenario_probabilities(
                model_name="TabDPT",
                scenario_name=scenario_name,
                subset_name="test",
                y_true=y_test,
                probabilities=probabilities
            )

            
            
            

            X_cal_scenario = X_cal_scenarios[scenario_name].copy()

            
            
            

        
            
            

            if scenario_name == "reference":
                probabilities_cal = proba_cal

            else:
                
                

                proba_cal_scenario_df = tabdpt_predictor.predict_proba(
                    X_cal_scenario
                )

                
                probabilities_cal = np.asarray(
                    proba_cal_scenario_df
                )[:, 1]


            
            

            save_scenario_probabilities(
                model_name="TabDPT",
                scenario_name=scenario_name,
                subset_name="calibration",
                y_true=y_cal,
                probabilities=probabilities_cal
            )

        
            
            

            if scenario_name == "reference":
                missing_prediction_time = prediction_time

            else:
                
                
                missing_prediction_time = (
                    time.perf_counter() - start_missing_prediction
                )


            
            
            metrics = calculate_metrics(
                y_test,
                probabilities
            )

            
            row = {
                "model": "TabDPT",
                "scenario": scenario_name,
                "prediction_time_seconds": missing_prediction_time,
                **metrics
            }

            tabdpt_missing_results.append(row)

            
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            print(
                f"Prediction time: {missing_prediction_time:.2f} s"
            )

            
            model_statuses.append({
                "model": "TabDPT",
                "scenario": scenario_name,
                "status": "scenario_success",
                "error": ""
            })
            save_checkpoint_missingness("tabdpt", tabdpt_missing_results)

        except Exception as scenario_error:
            
            
            del missingness_probabilities[probability_checkpoint_length:]
            print(f"\nERROR TabDPT, scenario {scenario_name}:")
            print(type(scenario_error).__name__, scenario_error)
            model_statuses.append({
                "model": "TabDPT",
                "scenario": scenario_name,
                "status": "scenario_failed",
                "error": f"{type(scenario_error).__name__}: {scenario_error}"
            })
            save_checkpoint_missingness("tabdpt", tabdpt_missing_results)
            continue

    
    tabdpt_missing_df = pd.DataFrame(
        tabdpt_missing_results
    )

    
    tabdpt_missing_df.to_csv(
        RESULTS_DIR / "tabdpt_missingness_results.csv",
        index=False
    )

    print("\nTABDPT - MISSINGNESS EVALUATION COMPLETED")


    del tabdpt_predictor

    gc.collect()


except Exception as e:


    print("\nERROR TABDPT:")

    print(type(e).__name__)

    print(e)

    model_statuses.append({
        "model": "TabDPT",
        "status": "failed",
        "error": f"{type(e).__name__}: {e}"
    })

    gc.collect()


# ============================================================

# 19. CATBOOST

# ============================================================


print("\n" + "=" * 70)
print("10. CATBOOST")

print("=" * 70)
# ============================================================


print("\n" + "=" * 70)

print("10. CATBOOST")

print("=" * 70)


try:


    # CatBoost costruisce alberi in sequenza: ciascun nuovo albero corregge
    
    
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


    # ============================================================
    
    # ============================================================

    # CatBoost è già stato addestrato su X_train completo.
    
    
    

    # CatBoost gestisce nativamente i valori NaN numerici.
    

    
    catboost_missing_results = []

    
    

    for scenario_name, X_scenario in X_test_scenarios.items():

        
        
        probability_checkpoint_length = len(missingness_probabilities)
        try:
            print(f"\nCATBOOST - Scenario: {scenario_name}")

            
            start_missing_prediction = time.perf_counter()

        
            
            
            

            if scenario_name == "reference":
                probabilities = proba_test

            else:
                
                
                # CatBoost gestisce nativamente i valori NaN.

                probabilities = catboost_model.predict_proba(
                    X_scenario
                )[:, 1]

            # ========================================================
            
            # ========================================================

            
            
            

            save_scenario_probabilities(
                model_name="CatBoost",
                scenario_name=scenario_name,
                subset_name="test",
                y_true=y_test,
                probabilities=probabilities
            )

            
            
            X_cal_scenario = X_cal_scenarios[scenario_name]

            #
            
            

            if scenario_name == "reference":
                probabilities_cal = proba_cal

            else:
                
                
                

                probabilities_cal = catboost_model.predict_proba(
                    X_cal_scenario
                )[:, 1]

            
            
            

            save_scenario_probabilities(
                model_name="CatBoost",
                scenario_name=scenario_name,
                subset_name="calibration",
                y_true=y_cal,
                probabilities=probabilities_cal
            )

        
            
            # misurato nel blocco originale di CatBoost.

            if scenario_name == "reference":
                missing_prediction_time = prediction_time

            else:
                
                

                missing_prediction_time = (
                    time.perf_counter() - start_missing_prediction
                )


            
            metrics = calculate_metrics(
                y_test,
                probabilities
            )

            
            row = {
                "model": "CatBoost",
                "scenario": scenario_name,
                "prediction_time_seconds": missing_prediction_time,
                **metrics
            }

            
            catboost_missing_results.append(row)

            
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            print(
                f"Prediction time: {missing_prediction_time:.2f} s"
            )

            
            model_statuses.append({
                "model": "CatBoost",
                "scenario": scenario_name,
                "status": "scenario_success",
                "error": ""
            })
            save_checkpoint_missingness("catboost", catboost_missing_results)

        except Exception as scenario_error:
            
            
            del missingness_probabilities[probability_checkpoint_length:]
            print(f"\nERROR CatBoost, scenario {scenario_name}:")
            print(type(scenario_error).__name__, scenario_error)
            model_statuses.append({
                "model": "CatBoost",
                "scenario": scenario_name,
                "status": "scenario_failed",
                "error": f"{type(scenario_error).__name__}: {scenario_error}"
            })
            save_checkpoint_missingness("catboost", catboost_missing_results)
            continue

    
    catboost_missing_df = pd.DataFrame(
        catboost_missing_results
    )

    
    catboost_missing_df.to_csv(
        RESULTS_DIR / "catboost_missingness_results.csv",
        index=False
    )

    print("\nCATBOOST - MISSINGNESS EVALUATION COMPLETED")


    del catboost_model

    gc.collect()


except Exception as e:


    print("\nERROR CATBOOST:")

    print(type(e).__name__)

    print(e)

    model_statuses.append({
        "model": "CatBoost",
        "status": "failed",
        "error": f"{type(e).__name__}: {e}"
    })

    gc.collect()


# ============================================================

# 20. MLP

# ============================================================


print("\n" + "=" * 70)

print("11. MLP")

print("=" * 70)


try:


    
    
    # quei parametri, senza calcolarli su calibration/test (data leakage).


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


    # ============================================================
    
    # ============================================================

    
    
    
    

    # L'MLP non può ricevere direttamente valori NaN.
    
    

    
    mlp_missing_results = []

    
    for scenario_name, X_scenario in X_test_scenarios.items():

        
        
        probability_checkpoint_length = len(missingness_probabilities)
        try:
            print(f"\nMLP - Scenario: {scenario_name}")

            
            
            
            start_missing_prediction = time.perf_counter()

            
            
            
            X_scenario_imputed = pd.DataFrame(
                imputer.transform(X_scenario),
                columns=X_scenario.columns,
                index=X_scenario.index
            )

        
            
            
            

            if scenario_name == "reference":
                probabilities = proba_test

            else:
                
                
                

                probabilities = mlp_model.predict_proba(
                    X_scenario_imputed
                )[:, 1]


            # ========================================================
            
            # ========================================================

            
            
            

            save_scenario_probabilities(
                model_name="MLP",
                scenario_name=scenario_name,
                subset_name="test",
                y_true=y_test,
                probabilities=probabilities
            )

            
            
            
            X_cal_scenario = X_cal_scenarios[scenario_name]

            # L'MLP non può ricevere direttamente valori NaN.
            
            
            

            X_cal_scenario_imputed = pd.DataFrame(
                imputer.transform(X_cal_scenario),
                columns=X_cal_scenario.columns,
                index=X_cal_scenario.index
            )

            #
            
            

            if scenario_name == "reference":
                probabilities_cal = proba_cal

            else:
                
                
                # Non ripetiamo l'addestramento dell'MLP.

                probabilities_cal = mlp_model.predict_proba(
                    X_cal_scenario_imputed
                )[:, 1]


            
            

            save_scenario_probabilities(
                model_name="MLP",
                scenario_name=scenario_name,
                subset_name="calibration",
                y_true=y_cal,
                probabilities=probabilities_cal
            )

        
            
            

            if scenario_name == "reference":
                missing_prediction_time = prediction_time

            else:
                
                
                

                missing_prediction_time = (
                    time.perf_counter() - start_missing_prediction
                )


            
            metrics = calculate_metrics(
                y_test,
                probabilities
            )

            
            row = {
                "model": "MLP",
                "scenario": scenario_name,
                "prediction_time_seconds": missing_prediction_time,
                **metrics
            }

            
            mlp_missing_results.append(row)

            
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            print(
                f"Prediction time: {missing_prediction_time:.2f} s"
            )

            
            model_statuses.append({
                "model": "MLP",
                "scenario": scenario_name,
                "status": "scenario_success",
                "error": ""
            })
            save_checkpoint_missingness("mlp", mlp_missing_results)

        except Exception as scenario_error:
            
            
            del missingness_probabilities[probability_checkpoint_length:]
            print(f"\nERROR MLP, scenario {scenario_name}:")
            print(type(scenario_error).__name__, scenario_error)
            model_statuses.append({
                "model": "MLP",
                "scenario": scenario_name,
                "status": "scenario_failed",
                "error": f"{type(scenario_error).__name__}: {scenario_error}"
            })
            save_checkpoint_missingness("mlp", mlp_missing_results)
            continue

    
    mlp_missing_df = pd.DataFrame(
        mlp_missing_results
    )

    
    mlp_missing_df.to_csv(
        RESULTS_DIR / "mlp_missingness_results.csv",
        index=False
    )

    print("\nMLP - MISSINGNESS EVALUATION COMPLETED")


    del mlp_model

    gc.collect()


except Exception as e:


    print("\nERROR MLP:")

    print(type(e).__name__)

    print(e)

    model_statuses.append({
        "model": "MLP",
        "status": "failed",
        "error": f"{type(e).__name__}: {e}"
    })

    gc.collect()


# ============================================================

# 21. SAVE RESULTS

# ============================================================


print("\n" + "=" * 70)

print("21. SALVATAGGIO RISULTATI")

print("=" * 70)


results_df = pd.DataFrame(

    results

)


results_df.to_csv(

    RESULTS_DIR

    / "sonar_reference_results.csv",

    index=False

)


probability_results.to_csv(

    RESULTS_DIR

    / "sonar_reference_probabilities.csv",

    index=False

)


# ============================================================

# ============================================================


#


#


if len(missingness_probabilities) > 0:

    
    
    missingness_probabilities_df = pd.concat(
        missingness_probabilities,
        ignore_index=True
    )

    
    missingness_probabilities_df.to_csv(
        RESULTS_DIR / "sonar_missingness_probabilities.csv",
        index=False
    )

    print("\nPROBABILITÀ MISSINGNESS SALVATE")

    print(
        "Numero totale di predizioni:",
        len(missingness_probabilities_df)
    )

    print(
        "File:",
        RESULTS_DIR / "sonar_missingness_probabilities.csv"
    )

else:

    
    
    print(
        "\nATTENZIONE: nessuna probabilità di missingness da salvare."
    )


status_df = pd.DataFrame(model_statuses)


status_df.to_csv(

    RESULTS_DIR

    / "sonar_model_status.csv",

    index=False

)


# ============================================================

# 22. FINAL RESULTS TABLE

# ============================================================


print("\n" + "=" * 70)

print("RISULTATI FINALI")

print("=" * 70)


if len(results_df) > 0:


    print(

        results_df.to_string(

            index=False

        )

    )


else:


    print(

        "Nessun modello ha prodotto risultati."

    )


# ============================================================

# 23. END

# ============================================================


print("\n" + "=" * 70)

print("PIPELINE COMPLETATA")

print("=" * 70)


print(

    "\nRisultati salvati in:"

)


print(

    RESULTS_DIR.resolve()

)


print(

    "\nFile creati:"

)


print(

    "- sonar_split.csv"

)


print(

    "- sonar_reference_results.csv"

)


print(

    "- sonar_reference_probabilities.csv"

)


print(

    "- sonar_model_status.csv"

)


print("- sonar_missingness_probabilities.csv (se esistono predizioni riuscite)")
print("- tabpfn/tabicl/tabdpt/catboost/mlp_missingness_results.csv")

print(
    "\nBenchmark missingness completato (controllare i CSV degli scenari falliti)."
)


