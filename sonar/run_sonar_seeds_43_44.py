# ============================================================
# SONAR - REPEATED EXPERIMENTS WITH SEEDS 43 AND 44
# Runs one seed at a time and checks output CSV completeness.
# Does not modify previously generated seed 42 results.
# ============================================================

from pathlib import Path
import subprocess
import sys
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS = {"TabPFN-v2", "TabICLv2", "TabDPT", "CatBoost", "MLP"}
SCENARIOS = {"reference", "cell_10", "cell_30", "cell_50",
             "feature_10", "feature_30", "feature_50"}


def run_script(filename):
    print(f"\n{'=' * 64}\nRUNNING: {filename}\n{'=' * 64}", flush=True)
    subprocess.run([sys.executable, str(BASE_DIR / filename)], cwd=BASE_DIR, check=True)


def check_probabilities(seed):
    folder = BASE_DIR / "results_sonar" / f"seed_{seed}"
    file = folder / "sonar_missingness_probabilities.csv"
    if not file.is_file():
        print(f"WARNING: missing file {file}")
        return False
    data = pd.read_csv(file)
    required_cols = {"model", "scenario", "subset", "observation_id", "y_true", "probability"}
    if not required_cols.issubset(data.columns):
        print("WARNING: missing columns in probabilities CSV.")
        return False

    missing = []
    for model in sorted(MODELS):
        for scenario in sorted(SCENARIOS):
            for subset in ("calibration", "test"):
                rows = data.loc[(data["model"] == model)
                                & (data["scenario"] == scenario)
                                & (data["subset"] == subset)]
                if rows.empty or rows["observation_id"].duplicated().any():
                    missing.append((model, scenario, subset))
    if missing:
        print(f"\nWARNING: seed {seed} incompleto: {len(missing)} missing or invalid groups.")
        for item in missing:
            print(" -", " / ".join(item))
        print("Check sonar_model_status.csv before running another seed.")
        return False
    print(f"\nSeed {seed}: 5 models x 7 scenarios x 2 subsets present.")
    return True


def main():
    if not (BASE_DIR / "sonar.all-data").is_file():
        raise FileNotFoundError("Place sonar.all-data in the same directory as these scripts.")
    for seed in (43, 44):
        folder = BASE_DIR / "results_sonar" / f"seed_{seed}"
        if (folder / "sonar_missingness_probabilities.csv").exists():
            answer = input(
                f"\nNel seed {seed} already has results. RE-RUN and OVERWRITE? [y/N]: "
            ).strip().lower()
            if answer != "y":
                print("Stopped to avoid overwriting existing results.")
                return

        run_script(f"sonar_models_seed_{seed}.py")
        if not check_probabilities(seed):
            print(f"\nStopped after seed {seed}: the next repeat will not run automatically.")
            return
        run_script(f"sonar_temperature_scaling_seed_{seed}.py")

    print("\nCOMPLETED: seeds 43 and 44, including temperature scaling and saved plots.")
    print("Original seed 42 results were not modified.")


if __name__ == "__main__":
    main()
