"""Regression checks for confirmed audit defects; no Spark JVM required."""
import importlib.util
import os
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get('PACE_REVIEW_SOURCE', Path(__file__).resolve().parents[1]))

def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class Frame:
    def __init__(self, pdf): self.pdf = pdf
    def toPandas(self): return self.pdf.copy()
    def limit(self, count): return Frame(self.pdf.head(count))

class Session:
    def createDataFrame(self, pdf): return Frame(pdf)

def test_empty_utility_is_not_a_measured_auc():
    Utility = module('review_utility', 'src/pace/silver/utilityassessment.py').UtilityAssessment
    with pytest.raises(ValueError):
        Utility({'label_column': 'y'})._train_eval(pd.DataFrame({'x': [None], 'y': [1]}), 'y')

def test_unknown_mechanism_cannot_silently_release_data():
    Engine = module('review_anon', 'src/pace/silver/anonymization.py').AnonymizationEngine
    with pytest.raises(ValueError):
        Engine({'technique': 'typo'}).anonymize(Frame(pd.DataFrame({'x': [1, 2]})), Session())

@pytest.mark.parametrize('technique', ['ldiversity', 'tcloseness'])
def test_structural_mechanism_requires_sensitive_attributes(technique):
    Engine = module('review_anon', 'src/pace/silver/anonymization.py').AnonymizationEngine
    with pytest.raises(ValueError):
        Engine({'technique': technique, 'quasi_identifiers': ['x'], 'k': 5}).anonymize(
            Frame(pd.DataFrame({'x': [1, 2]})), Session())

def test_inactive_epsilon_cannot_change_structural_score():
    Silver = module('review_layers', 'src/pace/metrics/layermetrics.py').SilverMetrics
    base = {'technique': 'kanonymity', 'k': 5, 'risk': .2, 'causal_validity': 'PASS'}
    assert Silver().calculate(dict(base, epsilon=.1)) == Silver().calculate(dict(base, epsilon=5.))

@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, 1.1, None])
def test_invalid_component_cannot_produce_readiness(bad):
    Score = module('review_score', 'src/pace/metrics/pacescore.py').PACEScore
    with pytest.raises(ValueError):
        Score({}).calculate(bad, .9, .9)

def test_invalid_weights_rejected():
    Score = module('review_score', 'src/pace/metrics/pacescore.py').PACEScore
    with pytest.raises(ValueError):
        Score({'weights': {'bronze': 1, 'silver': 1, 'gold': 1}}).calculate(.9, .9, .9)

def test_missing_fairness_group_is_not_perfect_fairness():
    Metrics = module('review_fairness', 'src/pace/gold/fairnessmetrics.py').FairnessMetrics
    config = {'protected_attribute': 'group', 'label_column': 'y', 'favorable_label': 1,
              'privileged_groups': [{'group': 'A'}], 'unprivileged_groups': [{'group': 'B'}]}
    report = Metrics(config).calculate(Frame(pd.DataFrame({'group': ['A'] * 20, 'y': [0, 1] * 10})))
    assert report['statistical_parity_difference'] is None
    assert report['disparate_impact'] is None
    assert 'error' in report
