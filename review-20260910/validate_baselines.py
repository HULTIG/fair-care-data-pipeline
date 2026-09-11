"""Independent diagnostic baselines; these are NOT PACE pipeline results.

Run from any directory with --source /path/to/source --output /new/output/dir.
Uses explicit predictors, train-only preprocessing, held-out predictions and
five seeded splits. No raw records or identifiers are written to the report.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

COMPAS = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
          'priors_count', 'c_charge_degree', 'sex']
NIJ = ['Gender', 'Age_at_Release', 'Residence_PUMA', 'Gang_Affiliated',
       'Supervision_Risk_Score_First', 'Supervision_Level_First',
       'Education_Level', 'Dependents', 'Prison_Offense', 'Prison_Years',
       'Prior_Arrest_Episodes_Felony', 'Prior_Arrest_Episodes_Misd',
       'Prior_Arrest_Episodes_Violent', 'Prior_Arrest_Episodes_Property',
       'Prior_Arrest_Episodes_Drug', 'Prior_Arrest_Episodes_DVCharges',
       'Prior_Arrest_Episodes_GunCharges', 'Prior_Conviction_Episodes_Felony',
       'Prior_Conviction_Episodes_Misd', 'Prior_Conviction_Episodes_Viol',
       'Prior_Conviction_Episodes_Prop', 'Prior_Conviction_Episodes_Drug',
       'Prior_Revocations_Parole', 'Prior_Revocations_Probation']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_data(source, name, cfg):
    path = source / cfg['raw_path']
    df = pd.read_csv(path, header=0 if cfg.get('has_header', True) else None,
                     names=cfg.get('column_names'), sep=cfg.get('delimiter', ','),
                     low_memory=False)
    n_raw = len(df)
    # Exact duplicate source records must never straddle train and test.
    df = df.drop_duplicates().reset_index(drop=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().replace({'?': np.nan, '': np.nan})
    target = cfg['label_column']
    if name == 'compas':
        features, group, privileged, unprivileged = COMPAS, df['race'], 'Caucasian', 'African-American'
        mapping = {0: 1, 1: 0}  # Favorable class: no two-year recidivism.
    elif name == 'adult':
        features = [c for c in df if c not in [target, 'sex', 'fnlwgt']]
        group, privileged, unprivileged = df['sex'], 'Male', 'Female'
        mapping = {'>50K': 1, '<=50K': 0}
    elif name == 'german':
        features = [c for c in df if c not in [target, 'personal_status_sex']]
        group = df['personal_status_sex'].map({'A91': 'male', 'A93': 'male', 'A94': 'male',
                                              'A92': 'female', 'A95': 'female'})
        privileged, unprivileged, mapping = 'male', 'female', {1: 1, 2: 0}
    else:
        features, group, privileged, unprivileged = NIJ, df['Race'], 'WHITE', 'BLACK'
        # Local NIJ CSV uses Yes/No; also accept the publisher's boolean encoding.
        mapping = {'No': 1, 'Yes': 0, False: 1, True: 0, 'False': 1, 'True': 0}
    y = df[target].map(mapping)
    if y.isna().any() or group.isna().any():
        raise ValueError(f'{name}: unknown or missing outcome/protected group; do not impute these')
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f'{name}: missing allowlisted features {sorted(missing)}')
    if not set([privileged, unprivileged]).issubset(set(group)):
        raise ValueError(f'{name}: configured comparison groups not present')
    return df[features], y.astype(int), group, privileged, unprivileged, {
        'source_sha256': digest(path), 'raw_rows': n_raw, 'evaluated_rows': len(df),
        'exact_duplicates_removed': n_raw - len(df), 'features': features,
        'target': target, 'favorable_class': 'encoded as 1',
        'privileged_group': privileged, 'unprivileged_group': unprivileged,
        'favorable_count': int(y.sum()), 'protected_group_counts': group.value_counts().to_dict(),
    }

def evaluate(X, y, group, privileged, unprivileged, seed):
    strata = y.astype(str) + ':' + group.astype(str)
    if strata.value_counts().min() < 2:
        strata = y
    train, test = train_test_split(np.arange(len(y)), test_size=.3, random_state=seed, stratify=strata)
    numeric = list(X.select_dtypes(include=[np.number]).columns)
    categorical = [c for c in X if c not in numeric]
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical)
    ])
    model = Pipeline([('preprocess', preprocess),
                      ('classifier', LogisticRegression(solver='liblinear', max_iter=2000, random_state=seed))])
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(X.iloc[train], y.iloc[train])
    probs = model.predict_proba(X.iloc[test])[:, list(model.classes_).index(1)]
    pred = (probs >= .5).astype(int)
    yt, gt = y.iloc[test].to_numpy(), group.iloc[test].to_numpy()
    rates = {}
    for label, value in [('privileged', privileged), ('unprivileged', unprivileged)]:
        mask = gt == value
        positive = mask & (yt == 1)
        negative = mask & (yt == 0)
        if not positive.any() or not negative.any():
            raise ValueError('Undefined group metric: each test group must have both outcome classes')
        rates[label] = {'n': int(mask.sum()), 'positive_outcomes': int(positive.sum()),
                        'selection_rate': float(pred[mask].mean()),
                        'true_positive_rate': float(pred[positive].mean()),
                        'false_positive_rate': float(pred[negative].mean())}
    return {'seed': seed, 'train_n': len(train), 'test_n': len(test),
            'split_sha256': hashlib.sha256(np.asarray(test, dtype='<i8').tobytes()).hexdigest(),
            'auc': float(roc_auc_score(yt, probs)),
            'balanced_accuracy': float(balanced_accuracy_score(yt, pred)),
            'dpd': rates['unprivileged']['selection_rate'] - rates['privileged']['selection_rate'],
            'eod': rates['unprivileged']['true_positive_rate'] - rates['privileged']['true_positive_rate'],
            'predicted_favorable_rate': float(pred.mean()), 'group_metrics': rates,
            'fit_and_evaluation_seconds': time.perf_counter() - started}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', default='42,43,44,45,46')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = yaml.safe_load((args.source / 'configs/default.yaml').read_text())['datasets']
    report = {'kind': 'independent_baseline_diagnostic_not_pipeline_results',
              'script_sha256': digest(Path(__file__)), 'python': platform.python_version(),
              'platform': platform.platform(),
              'packages': {p: importlib.metadata.version(p) for p in ['pandas', 'numpy', 'scikit-learn', 'PyYAML']},
              'protocol': {'test_fraction': .3, 'preprocessing': 'fit on training partition only',
                           'protected_attribute': 'excluded from model inputs; retained for evaluation',
                           'threshold': .5, 'std_ddof': 1, 'hardware_scope': 'single Python process'},
              'datasets': {}}
    for name in ['compas', 'adult', 'german', 'nij']:
        X, y, group, privileged, unprivileged, data_info = load_data(args.source, name, configs[name])
        runs = [evaluate(X, y, group, privileged, unprivileged, int(seed)) for seed in args.seeds.split(',')]
        aggregate = {metric: {'mean': float(np.mean([r[metric] for r in runs])),
                              'sample_std': float(np.std([r[metric] for r in runs], ddof=1))}
                     for metric in ['auc', 'balanced_accuracy', 'dpd', 'eod']}
        report['datasets'][name] = {'data': data_info, 'runs': runs, 'summary': aggregate}
        print(name, json.dumps(aggregate), flush=True)
    (args.output / 'baseline-diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
