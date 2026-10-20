"""Aggrega SONAR seed 42, 43, 44 senza addestrare modelli.
"""
from pathlib import Path
import csv
from collections import defaultdict
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results_sonar'
INPUTS = {
    42: RESULTS / 'sonar_temperature_scaling_results.csv',
    43: RESULTS / 'seed_43' / 'sonar_temperature_scaling_results.csv',
    44: RESULTS / 'seed_44' / 'sonar_temperature_scaling_results.csv',
}
METRICS = ['AUROC', 'Balanced_Accuracy', 'Log_Loss', 'Brier_Score', 'ECE']
SCENARIOS = ['reference', 'cell_10', 'cell_30', 'cell_50', 'feature_10', 'feature_30', 'feature_50']
METHODS = ['Uncalibrated', 'Reference_Calibration', 'Missingness_Augmented_Calibration']
MODELS = ['TabPFN-v2', 'TabICLv2', 'TabDPT', 'CatBoost', 'MLP']

def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def main():
    combined = []
    expected = {(m, s, c) for m in MODELS for s in SCENARIOS for c in METHODS}
    for seed, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f'Seed {seed}: file mancante: {path}')
        with path.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        actual = [(r['model'], r['scenario'], r['method']) for r in rows]
        if len(actual) != len(set(actual)):
            raise ValueError(f'Seed {seed}: combinazioni duplicate')
        if set(actual) != expected:
            missing = sorted(expected - set(actual))
            extra = sorted(set(actual) - expected)
            raise ValueError(f'Seed {seed}: dati incompleti o inattesi. Mancanti: {missing[:5]}; extra: {extra[:5]}')
        for r in rows:
            for col in METRICS + ['temperature']:
                r[col] = float(r[col])
            r['n_test'] = int(r['n_test'])
            r['seed'] = seed
            combined.append(r)

    by_key = defaultdict(list)
    for r in combined:
        by_key[(r['model'], r['scenario'], r['method'])].append(r)
    summary = []
    for model in MODELS:
        for scenario in SCENARIOS:
            for method in METHODS:
                records = sorted(by_key[(model, scenario, method)], key=lambda x: x['seed'])
                entry = {'model': model, 'scenario': scenario, 'method': method,
                         'n_seeds': len(records), 'seeds': ','.join(str(r['seed']) for r in records)}
                for metric in METRICS:
                    values = [r[metric] for r in records]
                    entry[f'{metric}_mean'] = mean(values)
                    entry[f'{metric}_std'] = stdev(values) if len(values) > 1 else 0.0
                summary.append(entry)

    RESULTS.mkdir(exist_ok=True)
    detail_cols = ['seed', 'model', 'scenario', 'method', 'temperature', 'n_test'] + METRICS
    summary_cols = ['model', 'scenario', 'method', 'n_seeds', 'seeds'] + [f'{metric}_{suffix}' for metric in METRICS for suffix in ('mean', 'std')]
    write_csv(RESULTS / 'sonar_all_seeds_results.csv', combined, detail_cols)
    write_csv(RESULTS / 'sonar_mean_std_results.csv', summary, summary_cols)
    print(f'Completato: {len(combined)} righe individuali, {len(summary)} gruppi aggregati')
    print('Riepilogo:', RESULTS / 'sonar_mean_std_results.csv')

if __name__ == '__main__':
    main()
