# SONAR probability calibration using temperature scaling.
# Temperatures are estimated only from calibration predictions.
# Scores and reliability diagrams use test predictions only.
# All generated files stay inside the directory for this random seed.

# ============================================================

# ============================================================


# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    log_loss,
    brier_score_loss,
)


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_sonar" / "seed_43"
INPUT_FILE = RESULTS_DIR / "sonar_missingness_probabilities.csv"

OUTPUT_PROBABILITIES = RESULTS_DIR / "sonar_calibrated_probabilities.csv"
OUTPUT_METRICS = RESULTS_DIR / "sonar_temperature_scaling_results.csv"
OUTPUT_TEMPERATURES = RESULTS_DIR / "sonar_temperatures.csv"
OUTPUT_STATUS = RESULTS_DIR / "sonar_temperature_scaling_status.csv"

SCENARIOS = (
    "reference", "cell_10", "cell_30", "cell_50",
    "feature_10", "feature_30", "feature_50",
)
AUGMENTED_SCENARIOS = ("reference", "cell_10", "cell_30")
EPSILON = 1e-12
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


def expected_calibration_error(y_true, probabilities, n_bins=10):
    """Compute confidence-based expected calibration error (ECE)."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not np.any(mask):
            continue
        bin_accuracy = np.mean(predictions[mask] == y_true[mask])
        bin_confidence = np.mean(confidence[mask])
        ece += np.mean(mask) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def calculate_metrics(y_true, probabilities):
    """Cinque metriche coerenti con quelle dello script SONAR originale."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, probabilities),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, predictions),
        "Log_Loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "Brier_Score": brier_score_loss(y_true, probabilities),
        "ECE": expected_calibration_error(y_true, probabilities),
    }


def temperature_transform(probabilities, temperature):
    """p_cal = sigmoid(logit(p_original) / T), con T strettamente positivo."""
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return expit(logit(probabilities) / temperature)


def fit_temperature(calibration_probabilities, calibration_labels):
    """Sceglie T minimizzando la Log Loss sul solo calibration set."""
    p = np.asarray(calibration_probabilities, dtype=float)
    y = np.asarray(calibration_labels, dtype=int)
    if len(p) == 0 or len(p) != len(y):
        raise ValueError("Il calibration set e' vuoto o incoerente.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Etichette o probabilita' di calibration non valide.")

    # Ottimizzare log(T) assicura che T resti strettamente positivo.
    def objective(log_temperature):
        t = np.exp(log_temperature)
        p_calibrated = temperature_transform(p, t)
        return log_loss(y, p_calibrated, labels=[0, 1])

    optimization = minimize_scalar(
        objective,
        bounds=(np.log(MIN_TEMPERATURE), np.log(MAX_TEMPERATURE)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not optimization.success:
        raise RuntimeError(f"Ottimizzazione temperatura fallita: {optimization.message}")
    return float(np.exp(optimization.x))


def validate_group(frame, model, scenario, subset):
    """Verifica che ogni osservazione compaia una volta e abbia dati validi."""
    if frame.empty:
        raise ValueError(f"Mancano {model}, {scenario}, {subset}.")
    if frame["observation_id"].duplicated().any():
        raise ValueError(f"Indici duplicati: {model}, {scenario}, {subset}.")
    if frame[["y_true", "probability"]].isna().any().any():
        raise ValueError(f"Valori mancanti: {model}, {scenario}, {subset}.")
    p = frame["probability"].to_numpy(dtype=float)
    y = frame["y_true"].to_numpy(dtype=int)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(f"Probabilita' non valide: {model}, {scenario}, {subset}.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"Target non valido: {model}, {scenario}, {subset}.")
    return frame.sort_values("observation_id").reset_index(drop=True)


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"File non trovato: {INPUT_FILE}\n"
            "Prima esegui lo script SONAR per salvare le probabilita'."
        )

    data = pd.read_csv(INPUT_FILE)
    expected_columns = {
        "model", "scenario", "subset", "observation_id", "y_true", "probability"
    }
    missing_columns = expected_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing_columns)}")

    data = data.copy()
    data["scenario"] = data["scenario"].astype(str)
    data["subset"] = data["subset"].astype(str)
    models = data["model"].dropna().unique().tolist()
    if not models:
        raise ValueError("Non e' stato trovato alcun modello nel CSV.")

    all_probabilities = []
    all_metrics = []
    temperatures = []
    statuses = []

    print("\n=== TEMPERATURE SCALING SONAR ===")
    print(f"Input: {INPUT_FILE}")
    print("Modelli disponibili:", ", ".join(models))

    for model in models:
        model_data = data.loc[data["model"] == model]
        calibration_groups = {}
        test_groups = {}
        for scenario in SCENARIOS:
            for subset, destination in (
                ("calibration", calibration_groups), ("test", test_groups)
            ):
                group = model_data.loc[
                    (model_data["scenario"] == scenario)
                    & (model_data["subset"] == subset)
                ]
                if group.empty:
                    continue
                try:
                    destination[scenario] = validate_group(group, model, scenario, subset)
                except Exception as error:
                    statuses.append({
                        "model": model, "scenario": scenario, "method": "input",
                        "status": "invalid", "error": str(error),
                    })
                    print(f"ATTENZIONE: {error}")

        # Verifica completezza rispetto allo scenario reference dello stesso subset.
        for subset_name, groups in (("calibration", calibration_groups), ("test", test_groups)):
            if "reference" not in groups:
                continue
            reference_ids = set(groups["reference"]["observation_id"])
            for scenario, group in list(groups.items()):
                if set(group["observation_id"]) != reference_ids:
                    msg = f"Osservazioni incomplete per {model}, {scenario}, {subset_name}"
                    print("ATTENZIONE:", msg)
                    statuses.append({
                        "model": model, "scenario": scenario, "method": "input",
                        "status": "incomplete", "error": msg,
                    })
                    del groups[scenario]

        temperatures_for_model = {}
        calibration_recipes = {
            "Reference_Calibration": ("reference",),
            "Missingness_Augmented_Calibration": AUGMENTED_SCENARIOS,
        }
        for method, required_scenarios in calibration_recipes.items():
            if not all(s in calibration_groups for s in required_scenarios):
                missing = [s for s in required_scenarios if s not in calibration_groups]
                msg = "Scenari calibration mancanti: " + ", ".join(missing)
                print(f"{model} | {method}: {msg}")
                statuses.append({
                    "model": model, "scenario": "all", "method": method,
                    "status": "skipped", "error": msg,
                })
                continue
            training_data = pd.concat(
                [calibration_groups[s] for s in required_scenarios], ignore_index=True
            )
            try:
                temperature = fit_temperature(
                    training_data["probability"], training_data["y_true"]
                )
            except Exception as error:
                statuses.append({
                    "model": model, "scenario": "all", "method": method,
                    "status": "failed", "error": str(error),
                })
                continue
            temperatures_for_model[method] = temperature
            temperatures.append({
                "model": model, "method": method, "temperature": temperature,
                "calibration_scenarios": "+".join(required_scenarios),
                "n_calibration_predictions": len(training_data),
                "n_unique_calibration_observations": training_data["observation_id"].nunique(),
            })
            print(f"{model} | {method} | T = {temperature:.5f}")

        for scenario in SCENARIOS:
            if scenario not in test_groups:
                statuses.append({
                    "model": model, "scenario": scenario, "method": "all",
                    "status": "skipped", "error": "Predizioni test mancanti o non valide",
                })
                continue
            test_group = test_groups[scenario]
            probabilities_original = test_group["probability"].to_numpy(dtype=float)
            labels = test_group["y_true"].to_numpy(dtype=int)

            methods = {"Uncalibrated": (1.0, probabilities_original)}
            for method, temperature in temperatures_for_model.items():
                methods[method] = (
                    temperature, temperature_transform(probabilities_original, temperature)
                )

            for method, (temperature, predictions) in methods.items():
                try:
                    metric_values = calculate_metrics(labels, predictions)
                except Exception as error:
                    statuses.append({
                        "model": model, "scenario": scenario, "method": method,
                        "status": "failed", "error": str(error),
                    })
                    continue

                rows = test_group[["observation_id", "y_true"]].copy()
                rows["model"] = model
                rows["scenario"] = scenario
                rows["method"] = method
                rows["temperature"] = temperature
                rows["original_probability"] = probabilities_original
                rows["calibrated_probability"] = predictions
                all_probabilities.append(rows)
                all_metrics.append({
                    "model": model, "scenario": scenario, "method": method,
                    "temperature": temperature, "n_test": len(test_group),
                    **metric_values,
                })
                statuses.append({
                    "model": model, "scenario": scenario, "method": method,
                    "status": "success", "error": "",
                })

        # Scriviamo i risultati gia' disponibili a fine modello.
        
        pd.concat(all_probabilities, ignore_index=True).to_csv(
            OUTPUT_PROBABILITIES, index=False
        ) if all_probabilities else None
        pd.DataFrame(all_metrics).to_csv(OUTPUT_METRICS, index=False)
        pd.DataFrame(temperatures).to_csv(OUTPUT_TEMPERATURES, index=False)
        pd.DataFrame(statuses).to_csv(OUTPUT_STATUS, index=False)

    print("\n=== COMPLETATO ===")
    print(f"Metriche:       {OUTPUT_METRICS}")
    print(f"Probabilita':  {OUTPUT_PROBABILITIES}")
    print(f"Temperature:   {OUTPUT_TEMPERATURES}")
    print(f"Stati:         {OUTPUT_STATUS}")
    print("\nNOTA: l'ottimizzazione usa solo CALIBRATION; TEST serve solo alla valutazione.")
    print("Gli scenari aumentati contengono tre versioni delle stesse osservazioni.")


if __name__ == "__main__":
    main()


    # ============================================================
    
    # ============================================================

    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    
    PLOTS_DIR = RESULTS_DIR / "reliability_diagrams_summary"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    
    plot_data = pd.read_csv(OUTPUT_PROBABILITIES)

    

    
    N_BINS = 5

    # Ordine fisso degli scenari
    SCENARIO_ORDER = [
        "reference",
        "cell_10",
        "cell_30",
        "cell_50",
        "feature_10",
        "feature_30",
        "feature_50"
    ]

    # Metodi da confrontare
    METHODS = [
        "Uncalibrated",
        "Reference_Calibration",
        "Missingness_Augmented_Calibration"
    ]

    METHOD_LABELS = {
        "Uncalibrated": "Uncalibrated",
        "Reference_Calibration": "Reference Calibration",
        "Missingness_Augmented_Calibration": "Augmented Calibration"
    }

    
    for model_name in sorted(plot_data["model"].unique()):

        model_df = plot_data[
            plot_data["model"] == model_name
        ].copy()

        if model_df.empty:
            continue

        # 7 scenari -> layout 3x3 (gli ultimi 2 pannelli verranno nascosti)
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        axes = axes.flatten()

        for i, scenario_name in enumerate(SCENARIO_ORDER):

            ax = axes[i]

            scenario_df = model_df[
                model_df["scenario"] == scenario_name
            ].copy()

            
            if scenario_df.empty:
                ax.set_visible(False)
                continue

            # Diagonale perfetta
            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="black",
                linewidth=1.2,
                label="Perfect Calibration"
            )

            for method_name in METHODS:

                method_df = scenario_df[
                    scenario_df["method"] == method_name
                ].copy()

                if method_df.empty:
                    continue

                y_true = method_df["y_true"].to_numpy()
                probabilities = method_df["calibrated_probability"].to_numpy()

                
                # calibration_curve potrebbe restituire meno di N_BINS punti
                frac_pos, mean_pred = calibration_curve(
                    y_true,
                    probabilities,
                    n_bins=N_BINS,
                    strategy="uniform"
                )

                ax.plot(
                    mean_pred,
                    frac_pos,
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    label=METHOD_LABELS[method_name]
                )

            ax.set_title(scenario_name.replace("_", " ").title(), fontsize=11)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.grid(alpha=0.3)

            
            if i == 0:
                ax.legend(fontsize=8, loc="best")

        # Nascondiamo i pannelli inutilizzati (8° e 9°)
        for j in range(len(SCENARIO_ORDER), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"Reliability Diagrams - {model_name}",
            fontsize=16
        )

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        safe_model_name = (
            model_name.replace("/", "_")
            .replace(" ", "_")
        )

        fig.savefig(
            PLOTS_DIR / f"{safe_model_name}_summary.png",
            dpi=300,
            bbox_inches="tight"
        )

        fig.savefig(
            PLOTS_DIR / f"{safe_model_name}_summary.pdf",
            bbox_inches="tight"
        )

        plt.close(fig)

    print("\nRELIABILITY DIAGRAM RIEPILOGATIVI COMPLETATI")
    print("Grafici salvati in:", PLOTS_DIR.resolve())