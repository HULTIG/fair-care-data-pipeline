class BronzeMetrics:
    def calculate(self, metadata: dict) -> float:
        # SB = w1*Provenance + w2*PII + w3*Quality
        # Simplified
        provenance_score = 1.0 if metadata.get("provenance_complete") else 0.5
        pii_score = 1.0 if not metadata.get("pii_found") else 0.0
        quality_score = metadata.get("quality_score", 0.8)
        
        return (provenance_score + pii_score + quality_score) / 3

class SilverMetrics:
    def calculate(self, metadata: dict) -> float:
        # SS = w1*Anonymization + w2*Utility + w3*Causal
        anon_score = 1.0 # Assumed if process ran
        utility_score = metadata.get("utility_retention", 0.0)
        causal_score = 1.0 if metadata.get("causal_validity") == "PASS" else 0.5
        
        return (anon_score + utility_score + causal_score) / 3

class GoldMetrics:
    def calculate(self, metadata: dict) -> float:
        # SG = (Fairness + ModelUtility) / 2.0
        import math
        spd = metadata.get("statistical_parity_difference")
        
        # Handle NaN, None, or missing SPD
        if spd is None or (isinstance(spd, float) and math.isnan(spd)):
            # If fairness couldn't be calculated, use a neutral score
            fairness_score = 0.5
        else:
            # Good fairness if SPD is close to 0. Heavily penalize structural bias.
            fairness_score = 1.0 if abs(spd) <= 0.1 else max(0.0, 1.0 - 2 * abs(spd))
        
        model_utility = metadata.get("model_utility", 0.5)
        
        return (fairness_score + model_utility) / 2.0

