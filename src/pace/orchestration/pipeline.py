import argparse
import yaml
import json
import os
import hashlib
from pyspark.sql import SparkSession
from pace.bronze.ingestion import DataIngestion
from pace.bronze.piidetection import PIIDetection
from pace.bronze.audittrail import AuditTrail
from pace.silver.anonymization import AnonymizationEngine
from pace.silver.causalanalysis import CausalAnalyzer
from pace.gold.biasmitigation import BiasMitigator
from pace.gold.fairnessmetrics import FairnessMetrics
from pace.gold.featureengineering import FeatureEngineer
from pace.gold.embeddings import EmbeddingsGenerator
from pace.metrics.layermetrics import BronzeMetrics, SilverMetrics, GoldMetrics
from pace.metrics.pacescore import PACEScore
from pace.metrics.compliance import ComplianceCheck
from pace.evaluation.contract import EvaluationContract
from pace.evaluation.splitting import split_source_records
from pace.silver.utilityassessment import UtilityAssessment

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def _run_pipeline(dataset, config_or_path, output_dir, verbose=False, seed=42):
    """
    Run the PACE pipeline and return metrics.
    Used by experiment scripts.
    """
    if isinstance(config_or_path, dict):
        config = config_or_path
    else:
        config = load_config(config_or_path)
    dataset_config = config['datasets'].get(dataset)
    if not dataset_config:
        raise ValueError(f"Dataset {dataset} not found in config.")
    config.setdefault("seed", seed)
    config_checksum = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()
    input_path = dataset_config.get("raw_path")
    input_checksum = None
    if input_path and os.path.isfile(input_path):
        digest = hashlib.sha256()
        with open(input_path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        input_checksum = digest.hexdigest()

    import time
    run_id = config.get("run_id") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-seed{seed}"
    output_dir = os.path.join(output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Spark with an explicit execution target. Unbounded implicit
    # local mode can exhaust the pilot host and terminate the JVM without
    # giving Python a catchable exception.
    spark_builder = SparkSession.builder \
        .appName(f"PACE-{dataset}-{run_id}") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true") \
        .config("spark.sql.shuffle.partitions", os.environ.get("PACE_SPARK_SHUFFLE_PARTITIONS", "8"))
    spark_master = os.environ.get("SPARK_MASTER")
    if spark_master:
        spark_builder = spark_builder.master(spark_master)
    elif not os.environ.get("SPARK_TESTING"):
        spark_builder = spark_builder.master("local[2]")
    spark = spark_builder.getOrCreate()

    audit = AuditTrail(log_dir=os.path.join(output_dir, "logs"))
    
    start_total = time.time()
    
    # --- BRONZE LAYER ---
    if verbose: print("\n=== BRONZE LAYER ===")
    start_bronze = time.time()
    ingestion = DataIngestion(spark)
    bronze_df = ingestion.ingest(
        dataset_config['raw_path'], 
        dataset_config['bronze_path'], 
        dataset,
        has_header=dataset_config.get('has_header', True),
        column_names=dataset_config.get('column_names'),
        delimiter=dataset_config.get('delimiter', ','),
        drop_columns=dataset_config.get('drop_columns'),
        predictor_allowlist=dataset_config.get('predictor_allowlist'),
        required_columns=[dataset_config.get('label_column'), dataset_config.get('protected_attribute')],
        missing_value_tokens=dataset_config.get('missing_value_tokens', ['?', ' ?']),
        label_column=dataset_config.get('label_column'),
        allowed_label_values=dataset_config.get('allowed_label_values'),
        protected_attribute=dataset_config.get('protected_attribute'),
        required_group_values=dataset_config.get('required_group_values')
    )

    # Establish the paired source split before any privacy, generalization,
    # imputation, feature, or mitigation operation can inspect the full frame.
    source_pdf = bronze_df.toPandas()
    train_pdf, test_pdf, split_metadata = split_source_records(
        source_pdf,
        label_column=dataset_config['label_column'],
        seed=seed,
    )
    split_metadata.update({
        "dataset": dataset,
        "record_id_column": "_record_id",
        "source_row_count": int(len(source_pdf)),
    })
    with open(os.path.join(output_dir, "split.json"), "w") as handle:
        json.dump(split_metadata, handle, indent=2)
    train_source_df = spark.createDataFrame(train_pdf)
    test_source_df = spark.createDataFrame(test_pdf)
    
    pii_detector = PIIDetection(config.get('pii_detection', {}))
    pii_report = pii_detector.detect(bronze_df)
    audit.log_event("PII_DETECTION", pii_report)
    
    # Calculate a simple quality score based on non-null ratio
    total_count = bronze_df.count()
    quality_score = (bronze_df.dropna().count() / total_count) if total_count > 0 else 0.0
    
    bronze_metrics = BronzeMetrics()
    sb = bronze_metrics.calculate({
        "provenance_complete": audit.verify_provenance(), 
        "pii_found": any(r.get('recommendation') == 'REVIEW' for r in pii_report.values()),
        "quality_score": quality_score
    })
    if verbose: print(f"Bronze Score (SB): {sb}")
    audit.log_event("STAGE_STATUS", {"stage": "bronze", "status": "executed"})
    time_bronze = time.time() - start_bronze

    # --- SILVER LAYER ---
    if verbose: print("\n=== SILVER LAYER ===")
    start_silver = time.time()
    anon_config = config.get('anonymization', {}).copy()
    anon_config['quasi_identifiers'] = dataset_config.get('quasi_identifiers', [])
    anon_config['label_column'] = dataset_config.get('label_column')
    anon_config['sensitive_attributes'] = dataset_config.get('sensitive_attributes', [])
    anon_config['protected_attribute'] = dataset_config.get('protected_attribute')
    
    anon_config['seed'] = seed
    anonymizer = AnonymizationEngine(anon_config)
    silver_df, silver_meta = anonymizer.fit_transform(train_source_df, spark)
    silver_test_df, silver_test_meta = anonymizer.transform(test_source_df, spark, enforce_postconditions=False)
    silver_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(dataset_config['silver_path'])
    
    causal_config = dataset_config.copy()
    causal_analyzer = CausalAnalyzer(causal_config)
    causal_enabled = config.get('stages', {}).get('causal_screening', config.get('causal_validation', True))
    causal_report = causal_analyzer.analyze(silver_df) if causal_enabled else {
        'status': 'skipped', 'causal_validity': 'UNAVAILABLE', 'reason': 'stage disabled'
    }
    audit.log_event("STAGE_STATUS", {"stage": "causal_screening", "status": "executed" if causal_enabled else "skipped"})
    audit.log_event("CAUSAL_ANALYSIS", causal_report)
    
    silver_metrics = SilverMetrics()
    ss = silver_metrics.calculate({
        "technique": anon_config.get("technique"),
        "epsilon": anon_config.get("epsilon"),
        "k": anon_config.get("k"),
        "technique": silver_meta.get("technique", anon_config.get("technique", "kanonymity")),
        "risk": silver_meta.get("risk") if silver_meta.get("risk") is not None else 1.0,
        "causal_validity": causal_report.get("causal_validity", "FAIL")
    })
    if verbose: print(f"Silver Score (SS): {ss}")
    audit.log_event("STAGE_STATUS", {"stage": "silver", "status": "executed"})
    time_silver = time.time() - start_silver

    # --- GOLD LAYER ---
    if verbose: print("\n=== GOLD LAYER ===")
    start_gold = time.time()
    bias_mitigator = BiasMitigator(dataset_config)
    mitigation_enabled = config.get('stages', {}).get('mitigation', config.get('bias_mitigation', True))
    gold_df = bias_mitigator.mitigate(silver_df, spark) if mitigation_enabled else silver_df
    audit.log_event("STAGE_STATUS", {"stage": "mitigation", "status": "executed" if mitigation_enabled else "skipped"})
    
    feature_engineer = FeatureEngineer(config)
    gold_df, feature_report = feature_engineer.process(gold_df)
    gold_test_df, _ = feature_engineer.process(silver_test_df)
    
    embeddings_enabled = config.get('stages', {}).get('embeddings', False)
    if embeddings_enabled:
        embeddings_gen = EmbeddingsGenerator(dataset_config)
        gold_df = embeddings_gen.generate(gold_df, spark)
    audit.log_event("STAGE_STATUS", {"stage": "embeddings", "status": "executed" if embeddings_enabled else "skipped"})
    
    fairness_metrics = FairnessMetrics(dataset_config)
    utility_assessment = UtilityAssessment(dataset_config)
    held_out = utility_assessment.predict_held_out(gold_df, gold_test_df, seed=seed)
    if mitigation_enabled:
        bias_mitigator.last_report["weights_consumed"] = bool(
            getattr(utility_assessment, "last_fit", {}).get("weights_consumed")
        )
    prediction_path = os.path.join(output_dir, "held_out_predictions.jsonl")
    held_out.to_json(prediction_path, orient="records", lines=True)
    prediction_sha256 = hashlib.sha256(open(prediction_path, "rb").read()).hexdigest()
    eval_report = EvaluationContract(dataset_config).evaluate(
        held_out,
        outcome_column=dataset_config['label_column'],
        prediction_column='prediction',
        score_column='prediction_score',
        protected_attribute=dataset_config['protected_attribute'],
        privileged_group=dataset_config.get('privileged_groups', [{}])[0],
        unprivileged_group=dataset_config.get('unprivileged_groups', [{}])[0],
        favorable_label=dataset_config.get('favorable_label', 1),
    )
    fairness_report = dict(eval_report)
    fairness_report['statistical_parity_difference'] = eval_report.get('demographic_parity_difference')
    utility_report = {
        'roc_auc': eval_report.get('roc_auc'),
        'balanced_accuracy': eval_report.get('balanced_accuracy'),
        'predicted_favorable_rate': eval_report.get('predicted_favorable_rate'),
        # Retention is a ratio against a separately measured baseline. A single
        # held-out run has no such denominator, so it must remain unavailable.
        'utility_retention': None,
        'evaluation_population': eval_report.get('record_count'),
        'status': eval_report.get('status'),
    }
    audit.log_event("FAIRNESS_METRICS", fairness_report)
    audit.log_event("EVALUATION_PROVENANCE", {
        "split": split_metadata,
        "fit": getattr(utility_assessment, "last_fit", {}),
        "mitigation": getattr(bias_mitigator, "last_report", {}),
        "test_record_count": int(len(test_pdf)),
    })
    if "error" in fairness_report or fairness_report.get("statistical_parity_difference") is None:
        spark.stop()
        raise ValueError(f"Fairness evaluation failed; run is invalid: {fairness_report}")
    
    gold_metrics = GoldMetrics()
    sg = gold_metrics.calculate({
        "statistical_parity_difference": fairness_report.get("statistical_parity_difference"),
        "model_utility": utility_report.get("roc_auc")
    })
    if verbose: print(f"Gold Score (SG): {sg}")
    audit.log_event("STAGE_STATUS", {"stage": "gold", "status": "executed"})
    time_gold = time.time() - start_gold

    # --- COMPOSITE SCORE ---
    if verbose: print("\n=== PACE SCORE ===")
    scorer = PACEScore(config)
    final_score = scorer.calculate(sb, ss, sg)
    if verbose: print(f"Final Score: {final_score}")
    
    # --- GOVERNANCE ENFORCEMENT ---
    checker = ComplianceCheck(config.get('compliance', {}))
    # We pass the final_score to check compliance
    # Assuming checker.evaluate takes the PACE score or similar metrics
    # In this mock, we'll just implement the logic based on the status
    evidence_valid = (
        all(value is not None for value in (sb, ss, sg, utility_report.get('roc_auc')))
        and eval_report.get('status') == 'valid'
        and utility_report.get('evaluation_population') == split_metadata['test_count']
    )
    compliance_report = checker.check({'pii_found': any(r.get('recommendation') == 'REVIEW' for r in pii_report.values())})
    compliance_status = final_score.get('status', 'UNKNOWN')
    if not evidence_valid or compliance_status == 'AT RISK' or not compliance_report.get('compliant', True):
        if verbose: print("\n[CONTROL PLANE ACTION] Dataset Locked: Governance Threshold Not Met. Promotion Prevented.")
        final_score['locked'] = True
    else:
        if verbose: print("\n[CONTROL PLANE ACTION] Dataset Approved for Promotion.")
        final_score['locked'] = False
    
    # Add detailed metrics for experiments
    final_score['fairness'] = fairness_report
    final_score['utility'] = utility_report
    final_score['privacy'] = {
        'risk': silver_meta.get('risk'),
        'information_loss': None
    }
    final_score['anonymization'] = {
        'k': anon_config.get('k', 0),
        'epsilon': anon_config.get('epsilon', float('inf')),
        'technique': silver_meta.get('technique', anon_config.get('technique')),
        'rows_input': silver_meta.get('rows_input'),
        'rows_retained': silver_meta.get('rows_retained'),
        'rows_suppressed': silver_meta.get('rows_suppressed'),
        'test_rows_input': silver_test_meta.get('rows_input'),
        'test_rows_transformed': silver_test_meta.get('rows_retained'),
    }
    final_score['split'] = split_metadata
    final_score['seed'] = int(seed)
    final_score['fit'] = getattr(utility_assessment, 'last_fit', {})
    final_score['mitigation'] = getattr(bias_mitigator, 'last_report', {})
    final_score['prediction_artifact'] = prediction_path
    final_score['prediction_artifact_sha256'] = prediction_sha256
    final_score['evaluation'] = {
        'outcome_column': dataset_config['label_column'],
        'protected_attribute': dataset_config['protected_attribute'],
        'favorable_label': dataset_config.get('favorable_label', 1),
        'privileged_values': dataset_config.get('privileged_values'),
        'unprivileged_values': dataset_config.get('unprivileged_values'),
        'privileged_group': dataset_config.get('privileged_groups', [{}])[0],
        'unprivileged_group': dataset_config.get('unprivileged_groups', [{}])[0],
        'prediction_threshold': 0.5,
    }
    final_score['provenance'] = {
        'resolved_config_sha256': config_checksum,
        'input_data_sha256': input_checksum,
        'code_revision': os.environ.get('PACE_CODE_REVISION'),
        'dirty_state': os.environ.get('PACE_CODE_DIRTY'),
    }
    # Promotion is the only point at which a Gold release is written. A
    # rejected run cannot overwrite or expose the previous eligible artifact.
    published = False
    if not final_score['locked'] and evidence_valid:
        gold_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(dataset_config['gold_path'])
        published = True
        audit.log_event("PUBLICATION", {"status": "eligible", "artifact": dataset_config['gold_path']})
    else:
        audit.log_event("PUBLICATION", {"status": "withheld", "reason": "evidence or policy check failed"})

    # Include the complete publication operation in the reported runtime.
    time_total = time.time() - start_total
    final_score['runtimes'] = {
        'bronze': time_bronze,
        'silver': time_silver,
        'gold': time_gold,
        'total': time_total
    }

    # Save Summary
    summary_path = os.path.join(output_dir, f"{dataset}_metricssummary.json")
    os.makedirs(output_dir, exist_ok=True)
    final_score['run_id'] = run_id
    final_score['evidence_valid'] = evidence_valid
    final_score['compliance'] = compliance_report
    final_score['publication'] = {'status': 'eligible' if published else 'withheld'}
    with open(summary_path, 'w') as f:
        json.dump(final_score, f, indent=2)
        
    if verbose: print(f"Pipeline complete in {time_total:.2f}s. Results saved to {output_dir}")
    spark.stop()
    
    return final_score


def run_pipeline(dataset, config_or_path, output_dir, verbose=False, seed=42):
    """Run one immutable experiment and persist failures without stale output."""
    import copy
    import time
    config = copy.deepcopy(config_or_path) if isinstance(config_or_path, dict) else load_config(config_or_path)
    run_id = config.get("run_id") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-seed{seed}"
    config["run_id"] = run_id
    run_dir = os.path.join(output_dir, run_id)
    audit = AuditTrail(log_dir=os.path.join(run_dir, "logs"))
    try:
        return _run_pipeline(dataset, config, output_dir, verbose, seed)
    except Exception as error:
        audit.log_failure(run_id, "pipeline", error)
        failure_path = os.path.join(run_dir, "failure.json")
        os.makedirs(run_dir, exist_ok=True)
        with open(failure_path, "w") as handle:
            json.dump({"run_id": run_id, "status": "failed", "error_type": type(error).__name__, "error": str(error)}, handle, indent=2)
        try:
            active = SparkSession.getActiveSession()
            if active is not None:
                active.stop()
        finally:
            raise

def main():
    parser = argparse.ArgumentParser(description="PACE Pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset name (compas, adult, german, nij)")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_pipeline(args.dataset, args.config, args.output, args.verbose)

if __name__ == "__main__":
    main()
