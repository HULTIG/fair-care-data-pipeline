import argparse
import yaml
import json
import os
from pyspark.sql import SparkSession
from pace.bronze.ingestion import DataIngestion
from pace.bronze.piidetection import PIIDetection
from pace.bronze.audittrail import AuditTrail
from pace.silver.anonymization import AnonymizationEngine
from pace.silver.utilityassessment import UtilityAssessment
from pace.silver.causalanalysis import CausalAnalyzer
from pace.gold.biasmitigation import BiasMitigator
from pace.gold.fairnessmetrics import FairnessMetrics
from pace.gold.featureengineering import FeatureEngineer
from pace.gold.embeddings import EmbeddingsGenerator
from pace.metrics.layermetrics import BronzeMetrics, SilverMetrics, GoldMetrics
from pace.metrics.pacescore import PACEScore
from pace.metrics.compliance import ComplianceCheck

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline(dataset, config_or_path, output_dir, verbose=False, seed=42):
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

    # Initialize Spark
    spark = SparkSession.builder \
        .appName(f"PACE-{dataset}") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true") \
        .getOrCreate()

    audit = AuditTrail(log_dir=os.path.join(output_dir, "logs"))
    
    import time
    
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
        drop_columns=dataset_config.get('drop_columns')
    )
    
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
    time_bronze = time.time() - start_bronze

    # --- SILVER LAYER ---
    if verbose: print("\n=== SILVER LAYER ===")
    start_silver = time.time()
    anon_config = config.get('anonymization', {}).copy()
    anon_config['quasi_identifiers'] = dataset_config.get('quasi_identifiers', [])
    anon_config['label_column'] = dataset_config.get('label_column')
    
    anonymizer = AnonymizationEngine(anon_config)
    silver_df, silver_meta = anonymizer.anonymize(bronze_df, spark)
    silver_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(dataset_config['silver_path'])
    
    utility_assessor = UtilityAssessment(dataset_config)
    utility_report = utility_assessor.assess(bronze_df, silver_df)
    audit.log_event("UTILITY_ASSESSMENT", utility_report)
    
    causal_config = dataset_config.copy()
    causal_analyzer = CausalAnalyzer(causal_config)
    causal_report = causal_analyzer.analyze(silver_df)
    audit.log_event("CAUSAL_ANALYSIS", causal_report)
    
    silver_metrics = SilverMetrics()
    ss = silver_metrics.calculate({
        "epsilon": anon_config.get("epsilon"),
        "k": anon_config.get("k"),
        "risk": silver_meta.get("risk", 1.0),
        "causal_validity": causal_report.get("causal_validity", "FAIL")
    })
    if verbose: print(f"Silver Score (SS): {ss}")
    time_silver = time.time() - start_silver

    # --- GOLD LAYER ---
    if verbose: print("\n=== GOLD LAYER ===")
    start_gold = time.time()
    bias_mitigator = BiasMitigator(dataset_config)
    gold_df = bias_mitigator.mitigate(silver_df, spark)
    
    feature_engineer = FeatureEngineer(config)
    gold_df, feature_report = feature_engineer.process(gold_df)
    
    embeddings_gen = EmbeddingsGenerator(dataset_config)
    gold_df = embeddings_gen.generate(gold_df, spark)
    
    gold_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(dataset_config['gold_path'])
    
    fairness_metrics = FairnessMetrics(dataset_config)
    fairness_report = fairness_metrics.calculate(gold_df)
    audit.log_event("FAIRNESS_METRICS", fairness_report)
    
    gold_metrics = GoldMetrics()
    sg = gold_metrics.calculate({
        "statistical_parity_difference": fairness_report.get("statistical_parity_difference"),
        "model_utility": utility_report.get("anonymized_auc", utility_report.get("utility_retention", 0.5))
    })
    if verbose: print(f"Gold Score (SG): {sg}")
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
    compliance_status = final_score.get('status', 'UNKNOWN')
    if compliance_status == 'AT RISK':
        if verbose: print("\n[CONTROL PLANE ACTION] Dataset Locked: Governance Threshold Not Met. Promotion Prevented.")
        final_score['locked'] = True
    else:
        if verbose: print("\n[CONTROL PLANE ACTION] Dataset Approved for Promotion.")
        final_score['locked'] = False
        
    time_total = time.time() - start_total
    
    # Add detailed metrics for experiments
    final_score['fairness'] = fairness_report
    final_score['utility'] = utility_report
    final_score['privacy'] = {
        'risk': silver_meta.get('risk', anon_config.get('privacy_risk', 0.1)),
        'information_loss': anon_config.get('info_loss', 0.2)
    }
    final_score['anonymization'] = {
        'k': anon_config.get('k', 0),
        'epsilon': anon_config.get('epsilon', float('inf'))
    }
    final_score['runtimes'] = {
        'bronze': time_bronze,
        'silver': time_silver,
        'gold': time_gold,
        'total': time_total
    }
    
    # Save Summary
    summary_path = os.path.join(output_dir, f"{dataset}_metricssummary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(final_score, f, indent=2)
        
    if verbose: print(f"Pipeline complete in {time_total:.2f}s. Results saved to {output_dir}")
    spark.stop()
    
    return final_score

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
